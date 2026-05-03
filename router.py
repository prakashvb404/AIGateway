"""
AI Router — token-disciplined rewrite
--------------------------------------
Every model call sends only what that call actually needs.

Sonnet calls:
  - sonnet_plan      → last user turn only, 400 tok cap
  - sonnet_polish    → question + qwen answer only, 500 tok cap

Qwen3 calls:
  - rate_difficulty  → 1 message, no history, no tools, thinking OFF
  - qwen_execute     → trimmed history (last MAX_CTX_MESSAGES), thinking ON
"""

import asyncio
import copy
import hashlib
import hmac
import logging
import json
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Regex to strip <system-reminder> blocks from message content
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def sanitize_for_local(content: object) -> object:
    """Strip MCP/system-reminder noise from message content before sending to local model."""
    if isinstance(content, str):
        cleaned = _SYSTEM_REMINDER_RE.sub("", content).strip()
        return cleaned if cleaned else content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and not item.get("type"):
                    item = {**item, "text": sanitize_for_local(item["text"])}
                elif item.get("type") == "text":
                    item = {**item, "text": sanitize_for_local(item.get("text", ""))}
                else:
                    item = {k: sanitize_for_local(v) for k, v in item.items()}
            out.append(item)
        return out
    return content


# ── load config ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(_CONFIG_PATH, "r") as f:
        _CFG = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    raise RuntimeError(f"Failed to load config.json: {e}") from e

_CFG_MODELS = _CFG["models"]
_CFG_ROUTING = _CFG["routing"]
_CFG_TOKENS = _CFG["token_budgets"]
_CFG_PROMPTS = _CFG["prompts"]
_CFG_SERVER = _CFG["server"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _client.aclose()


app = FastAPI(lifespan=lifespan)

# ── metrics tracking ────────────────────────────────────────────────────────────
_metrics_lock = asyncio.Lock()
# Estimated Sonnet tokens saved per simple-tier question (plan ~400 + polish ~500)
SONNET_TOKENS_SAVED_PER_SIMPLE = 900

# ── per-model pricing ($ per 1M tokens) — loaded from config ───────────────────
MODEL_PRICING = {name: cfg["pricing"] for name, cfg in _CFG_MODELS.items()}
# Shorthand key for cloud model pricing (used across /stats and /dashboard)
SONNET_PRICING_KEY = _CFG_MODELS["cloud"]["name"]
MODEL_PRICING[SONNET_PRICING_KEY] = _CFG_MODELS["cloud"]["pricing"]

_metrics_data: dict = {
    "total_requests": 0,
    "calls": defaultdict(lambda: {"count": 0, "input_tokens": 0, "output_tokens": 0}),
    "conversations": 0,
    "tiers": {"simple": 0, "medium": 0, "hard": 0},
    "errors": 0,
    "tokens_saved": 0,
    # per-tier token tracking for savings calculation
    "tier_tokens": {
        "simple": {"input": 0, "output": 0},
        "medium": {"input": 0, "output": 0},
        "hard": {"input": 0, "output": 0},
    },
    "history": [],
    # per-model tracking
    "models": defaultdict(
        lambda: {
            "count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
        }
    ),
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate with better accuracy for code/JSON.

    Default heuristic: ~4 chars/token works for prose but breaks for
    code and JSON where tokens are shorter (keywords, symbols).
    We detect code-like content by looking for common patterns and
    adjust the divisor accordingly.
    """
    length = len(text)
    if length == 0:
        return 1

    # Detect code-like content: ratio of non-alpha chars to total
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / length if length > 0 else 0

    # Code/JSON has lots of symbols, short tokens → ~2.5 chars/token
    # Prose has longer words → ~4 chars/token
    # Mixed: interpolate between the two
    if alpha_ratio < 0.5:
        # Code-heavy (lots of { }, ;, etc.)
        chars_per_token = 2.5
    elif alpha_ratio < 0.7:
        # Mixed code/prose
        chars_per_token = 3.0
    else:
        # Prose-heavy
        chars_per_token = 4.0

    return max(1, int(length / chars_per_token))


async def record_call(call_type: str, input_text: str = "", output_text: str = ""):
    async with _metrics_lock:
        m = _metrics_data["calls"][call_type]
        m["count"] += 1
        if input_text:
            m["input_tokens"] += _estimate_tokens(input_text)
        if output_text:
            m["output_tokens"] += _estimate_tokens(output_text)


async def record_model_usage(
    model_name: str, input_tokens: int = 0, output_tokens: int = 0
):
    """Track per-model token usage and estimated cost.

    Only accumulates tokens; recalculates cost from the running totals.
    This avoids the bug where repeated calls re-add the same usage to cost.
    """
    async with _metrics_lock:
        md = _metrics_data["models"][model_name]
        md["count"] += 1
        md["input_tokens"] += input_tokens
        md["output_tokens"] += output_tokens
        # Recalculate cost from cumulative totals (not additively)
        pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        md["estimated_cost"] = round(
            (md["input_tokens"] / 1_000_000 * pricing["input"])
            + (md["output_tokens"] / 1_000_000 * pricing["output"]),
            6,
        )


async def record_request(tier: str, difficulty: int):
    async with _metrics_lock:
        _metrics_data["total_requests"] += 1
        _metrics_data["conversations"] += 1
        _metrics_data["tiers"][tier] = _metrics_data["tiers"].get(tier, 0) + 1
        if tier == "simple":
            _metrics_data["tokens_saved"] += SONNET_TOKENS_SAVED_PER_SIMPLE

        # Track per-tier token usage for savings calculation
        tier_tokens = _metrics_data["tier_tokens"][tier]
        # Estimate tokens used by this request (rough: simple=150, medium=2000, hard=3000)
        if tier == "simple":
            est_in, est_out = 150, 150
        elif tier == "medium":
            est_in, est_out = 2000, 800
        else:
            est_in, est_out = 3000, 1200
        tier_tokens["input"] += est_in
        tier_tokens["output"] += est_out

        # Compute cumulative savings for this point in time
        actual_cost = sum(
            md["estimated_cost"] for md in _metrics_data["models"].values()
        )
        sonnet_pricing = MODEL_PRICING.get(
            SONNET_PRICING_KEY, _CFG_MODELS["cloud"]["pricing"]
        )
        # Hypothetical: what if every tier went to Sonnet at full price
        tier_savings = {
            "simple": SONNET_TOKENS_SAVED_PER_SIMPLE,  # plan+polish tokens avoided
            "medium": 2500,  # typical medium query tokens sent to Sonnet
            "hard": 4200,  # typical hard query tokens sent to Sonnet
        }
        hypothetical = (
            sum(
                _metrics_data["tier_tokens"][t]["input"]
                + _metrics_data["tier_tokens"][t]["output"]
                for t in ("medium", "hard")
            )
            / 1_000_000
            * sonnet_pricing["input"]
        )
        hypothetical += (
            sum(
                tier_savings[t] * _metrics_data["tiers"].get(t, 0)
                for t in ("simple", "medium", "hard")
            )
            / 1_000_000
            * sonnet_pricing["output"]
        )
        cumulative_savings = round(max(0, hypothetical - actual_cost), 6)

        _metrics_data["history"].append(
            {
                "time": time.strftime("%H:%M:%S"),
                "epoch": time.time(),
                "tier": tier,
                "difficulty": difficulty,
                "savings": cumulative_savings,
            }
        )
        if len(_metrics_data["history"]) > 100:
            _metrics_data["history"] = _metrics_data["history"][-100:]


async def record_error():
    async with _metrics_lock:
        _metrics_data["errors"] += 1


# ── endpoints — loaded from config ─────────────────────────────────────────────
LOCAL_AI_URL = _CFG_MODELS["local"]["url"]
CLOUD_AI_URL = _CFG_MODELS["cloud"]["url"]
CLASSIFY_AI_URL = _CFG_MODELS["classify"]["url"]

# ── model names — loaded from config ───────────────────────────────────────────
CLOUD_MODEL = _CFG_MODELS["cloud"]["name"]
LOCAL_MODEL = _CFG_MODELS["local"]["name"]
CLASSIFY_MODEL = _CFG_MODELS["classify"]["name"]

# ── token budgets — loaded from config ─────────────────────────────────────────
CLASSIFY_MAX_TOKENS = _CFG_TOKENS["classify_max_tokens"]
PLAN_MAX_TOKENS = _CFG_TOKENS["plan_max_tokens"]
POLISH_MAX_TOKENS = _CFG_TOKENS["polish_max_tokens"]
EXECUTE_MAX_TOKENS = _CFG_TOKENS["execute_max_tokens"]
MAX_TOOL_RESULT_CHARS = _CFG_ROUTING["max_tool_result_chars"]

# ── context window trim — loaded from config ───────────────────────────────────
MAX_CTX_MESSAGES = _CFG_ROUTING["max_ctx_messages"]

# ── difficulty bands — loaded from config ──────────────────────────────────────
SIMPLE_MAX = _CFG_ROUTING["difficulty_bands"]["simple_max"]
MEDIUM_MAX = _CFG_ROUTING["difficulty_bands"]["medium_max"]
HARD_MAX = 10

# ── prompts — loaded from config ───────────────────────────────────────────────
_CLASSIFY_SYS = _CFG_PROMPTS["classify_sys"]
_PLAN_SYS = _CFG_PROMPTS["plan_sys"]
_POLISH_SYS = _CFG_PROMPTS["polish_sys"]
_WORKER_PREFIX = _CFG_PROMPTS["worker_prefix"]

# ── state ───────────────────────────────────────────────────────────────────────
_conv_state: dict[str, dict] = {}
_rating_locks: dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()  # guards _rating_locks creation — prevents two coroutines
# from racing into setdefault and getting different Lock objects

# ── shared HTTP client ───────────────────────────────────────────────────────────
_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def conv_key(messages: list) -> str:
    """
    Fingerprint based on user messages only, using HMAC-SHA256.

    Previously truncated to 200 chars per message which caused collisions
    between conversations with similar-looking logs or templates.
    Now hashes the full content so every conversation gets a unique key
    while keeping the output at a fixed 40-char hex length.
    """
    # Concatenate all user message content — full text, no truncation
    parts = (str(m.get("content", "")) for m in messages if m.get("role") == "user")
    raw = "|".join(parts)
    # HMAC with a fixed salt prevents adversarial collision attacks
    return hmac.new(b"conv-key-salt", raw.encode(), hashlib.sha256).hexdigest()[:40]


def trim_messages(messages: list, max_messages: int = MAX_CTX_MESSAGES) -> list:
    """
    Keep the system prompt + the last `max_messages` non-system messages.
    Prevents ballooning context on long conversations.
    """
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return system + rest[-max_messages:]


def strip_tool_messages(messages: list) -> list:
    """
    Remove tool-call/tool-result messages before sending to Sonnet.
    Sonnet doesn't execute tools here — it only plans or polishes.
    Tool JSON is pure noise for those calls.
    """
    return [
        m
        for m in messages
        if m.get("role") not in ("tool",)
        and not (m.get("role") == "assistant" and "tool_calls" in m)
    ]


def inject_system(messages: list, prefix: str) -> list:
    """Replace (or create) the system prompt.

    Old behavior prepended to existing system prompts on every call,
    causing instruction drift as worker prefixes and plans stacked up.
    Now it always replaces, keeping the system prompt clean and focused.
    """
    messages = list(messages)
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            messages[i] = {**m, "content": prefix}
            return messages
    return [{"role": "system", "content": prefix}] + messages


def apply_thinking(messages: list) -> list:
    # No /think injection needed — thinking is enabled via the "thinking" param in the body.
    return messages


def should_disable_thinking(messages: list) -> bool:
    """llama-server rejects thinking=enabled when conversation has pre-filled assistant responses."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content")
            if content is not None or "tool_calls" in m:
                return True
        if m.get("role") == "user":
            return False
    return False


def truncate_tool_results(messages: list) -> list:
    """Truncate oversized tool result content so it doesn't blow the context window."""
    out = []
    for m in messages:
        if m.get("role") == "tool":
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > MAX_TOOL_RESULT_CHARS:
                m = {**m, "content": content[:MAX_TOOL_RESULT_CHARS] + "\n…[truncated]"}
        out.append(m)
    return out


def has_tool_turn(messages: list) -> bool:
    """True if the most recent exchange contains a tool result."""
    for m in reversed(messages):
        if m.get("role") == "tool":
            return True
        if m.get("role") == "user":
            return False
    return False


def build_reply(
    content: str, difficulty: int, tier: str, plan_used: bool = True
) -> str:
    labels = {
        "simple": f"⚡ Qwen3 · difficulty {difficulty}/10",
        "medium": f"⚡ Qwen3 worker · {'🔴 Sonnet plan+polish' if plan_used else 'Qwen3 only'} · difficulty {difficulty}/10",
        "hard": f"⚡ Qwen3 worker · {'🔴 Sonnet plan+polish' if plan_used else 'Qwen3 only'} · difficulty {difficulty}/10",
    }
    return f"{content}\n\n---\n*{labels.get(tier, tier)}*"


# ── SSE / response helpers ───────────────────────────────────────────────────────


def _chunk(cid, created, model, delta, finish=None):
    return json.dumps(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def make_sse(content: str, model: str) -> StreamingResponse:
    cid, ts = "chatcmpl-proxy", int(time.time())

    def chunks():
        yield f"data: {_chunk(cid, ts, model, {'role': 'assistant', 'content': ''})}\n\n"
        yield f"data: {_chunk(cid, ts, model, {'content': content})}\n\n"
        yield f"data: {_chunk(cid, ts, model, {}, 'stop')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(chunks(), media_type="text/event-stream")


def make_tool_sse(data: dict) -> StreamingResponse:
    cid = data.get("id", "chatcmpl-proxy")
    created = data.get("created", int(time.time()))
    model = data.get("model", LOCAL_MODEL)
    msg = data["choices"][0].get("message", {})

    def chunks():
        yield f"data: {_chunk(cid, created, model, {'role': 'assistant', 'content': None})}\n\n"
        if "tool_calls" in msg:
            yield f"data: {_chunk(cid, created, model, {'tool_calls': msg['tool_calls']})}\n\n"
        yield f"data: {_chunk(cid, created, model, {}, 'tool_calls')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(chunks(), media_type="text/event-stream")


def make_json(base: dict, content: str) -> Response:
    out = copy.deepcopy(base)
    out["choices"][0]["message"]["content"] = content
    return Response(
        content=json.dumps(out).encode(), status_code=200, media_type="application/json"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model calls — each sends ONLY what it needs
# ══════════════════════════════════════════════════════════════════════════════


async def rate_difficulty(question: str, headers: dict) -> int:
    """
    Qwen3 classifier, 10-token cap.
    Input: single user message — no history, no tools.
    """
    body = {
        "model": CLASSIFY_MODEL,
        "max_completion_tokens": CLASSIFY_MAX_TOKENS,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": _CLASSIFY_SYS},
            {"role": "user", "content": question[:1000]},
        ],
    }
    log.info("🧠 CLASSIFY (%d chars): %.80s…", len(question), question)
    try:
        r = await _client.post(
            CLASSIFY_AI_URL,
            content=json.dumps(body).encode(),
            headers=headers,
            timeout=20.0,
        )
        r.raise_for_status()
        resp = r.json()
        text = resp["choices"][0]["message"]["content"].strip()

        # Track actual token usage from response — classify uses CLASSIFY_MODEL, not CLOUD_MODEL
        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                CLASSIFY_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        else:
            # Fallback: estimate tokens when model doesn't return usage
            input_text = json.dumps(body["messages"])
            output_text = text
            await record_model_usage(
                CLASSIFY_MODEL,
                input_tokens=_estimate_tokens(input_text),
                output_tokens=_estimate_tokens(output_text),
            )

        match = re.search(r"\d+", text)
        # empty response (model returned nothing) → safe default by question length
        if not text:
            score = 1 if len(question) < 20 else 5
            log.warning("🧠 CLASSIFY returned empty — heuristic default: %d", score)
            return score
        score = max(1, min(10, int(match.group()))) if match else 5
        log.info("🧠 SCORE: %d (raw: %r)", score, text)
        return score
    except Exception as e:
        log.warning("CLASSIFY ERROR: %s — default 5", e)
        return 5


async def sonnet_plan(question: str, messages: list, headers: dict) -> Optional[str]:
    """
    Sonnet planner.
    Input: system prompt + last user turn only (no tool noise, no old history).
    Cost: ~300-600 input tokens, <= 400 output tokens.
    """
    relevant = strip_tool_messages(messages)
    relevant = trim_messages(relevant, max_messages=4)
    plan_messages = inject_system(relevant, _PLAN_SYS)
    body = {
        "model": CLOUD_MODEL,
        "stream": False,
        "max_completion_tokens": PLAN_MAX_TOKENS,
        "messages": plan_messages,
    }
    try:
        r = await _client.post(
            CLOUD_AI_URL,
            content=json.dumps(body).encode(),
            headers=headers,
            timeout=180.0,
        )
        r.raise_for_status()
        resp = r.json()
        plan = resp["choices"][0]["message"]["content"].strip()

        # Track actual token usage from response
        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

        log.info("🔴 PLAN (%d chars): %.200s…", len(plan), plan)
        return plan
    except Exception as e:
        log.warning("PLAN ERROR: %s — type: %s", e, type(e).__name__)
        return None


async def qwen_execute(
    messages: list, body_json: dict, headers: dict
) -> tuple[Optional[Response], Optional[dict]]:
    """
    Qwen3 worker — thinking ON, trimmed context window.
    Tool history is preserved (needed for the execution loop).
    Context capped at MAX_CTX_MESSAGES to avoid runaway token growth.
    """
    trimmed = trim_messages(messages, MAX_CTX_MESSAGES)
    trimmed = truncate_tool_results(trimmed)
    # Strip MCP system-reminder noise that breaks llama-server
    trimmed = sanitize_for_local(trimmed)
    disable_thinking = should_disable_thinking(trimmed)
    local_body = {
        **body_json,
        "model": LOCAL_MODEL,
        "stream": False,
        "max_completion_tokens": EXECUTE_MAX_TOKENS,
        "messages": apply_thinking(trimmed),
        "thinking": {"type": "disabled"} if disable_thinking else {"type": "enabled"},
    }
    try:
        r = await _client.post(
            LOCAL_AI_URL,
            content=json.dumps(local_body).encode(),
            headers=headers,
            timeout=300.0,
        )
        r.raise_for_status()
        resp = r.json()

        # Qwen3 with thinking returns reasoning_content + empty content.
        # Promote reasoning_content → content for downstream compatibility,
        # but keep reasoning_content intact so the frontend can render
        # thinking blocks if desired.
        for choice in resp.get("choices", []):
            msg = choice.get("message", {})
            if not msg.get("content") and msg.get("reasoning_content"):
                msg["content"] = msg["reasoning_content"]

        # Track local model usage
        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                LOCAL_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        else:
            # Fallback: estimate tokens when model doesn't return usage
            input_text = json.dumps(local_body)
            output_text = json.dumps(resp.get("choices", []))
            await record_model_usage(
                LOCAL_MODEL,
                input_tokens=_estimate_tokens(input_text),
                output_tokens=_estimate_tokens(output_text),
            )

        return None, resp
    except Exception as e:
        log.error("EXECUTE ERROR: %s (url=%s, model=%s)", e, LOCAL_AI_URL, LOCAL_MODEL)
        return Response(content=json.dumps({"error": str(e)}), status_code=502), None


async def _sonnet_direct(
    messages: list, body_json: dict, headers: dict
) -> tuple[Optional[Response], Optional[dict]]:
    """Send the full conversation directly to Sonnet (used for hard/7-10)."""
    sonnet_body = {
        **body_json,
        "model": CLOUD_MODEL,
        "stream": False,
        "messages": messages,
    }
    try:
        r = await _client.post(
            CLOUD_AI_URL,
            content=json.dumps(sonnet_body).encode(),
            headers=headers,
            timeout=300.0,
        )
        r.raise_for_status()
        resp = r.json()

        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        else:
            input_text = json.dumps(sonnet_body)
            output_text = json.dumps(resp.get("choices", []))
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=_estimate_tokens(input_text),
                output_tokens=_estimate_tokens(output_text),
            )

        return None, resp
    except Exception as e:
        log.error("DIRECT SONNET ERROR: %s", e)
        return Response(content=json.dumps({"error": str(e)}), status_code=502), None


async def sonnet_polish(
    question: str, worker_reply: str, headers: dict
) -> Optional[str]:
    """
    Sonnet polisher.
    Input: question + Qwen3 answer ONLY — no conversation history, no tools.
    Cost: len(question) + len(answer) input tokens, <= 500 output tokens.
    """
    body = {
        "model": CLOUD_MODEL,
        "stream": False,
        "max_completion_tokens": POLISH_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": _POLISH_SYS},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nAnswer to improve:\n{worker_reply}",
            },
        ],
    }
    try:
        r = await _client.post(
            CLOUD_AI_URL,
            content=json.dumps(body).encode(),
            headers=headers,
            timeout=180.0,
        )
        r.raise_for_status()
        resp = r.json()
        polished = resp["choices"][0]["message"]["content"].strip()

        # Track actual token usage from response
        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        else:
            # Fallback: estimate tokens when model doesn't return usage
            input_text = (
                _POLISH_SYS
                + "\nQuestion:\n"
                + question
                + "\n\nAnswer to improve:\n"
                + worker_reply
            )
            output_text = polished
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=_estimate_tokens(input_text),
                output_tokens=_estimate_tokens(output_text),
            )

        log.info("🔴 POLISH (%d chars): %.200s…", len(polished), polished)
        return polished
    except Exception as e:
        log.warning(
            "POLISH ERROR: %s — type: %s — using Qwen3 as-is", e, type(e).__name__
        )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Request handler
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/metrics")
async def get_metrics():
    async with _metrics_lock:
        return JSONResponse(content=dict(_metrics_data))


@app.get("/stats")
async def get_stats():
    async with _metrics_lock:
        total_calls = sum(m["count"] for m in _metrics_data["calls"].values())
        total_input = sum(m["input_tokens"] for m in _metrics_data["calls"].values())
        total_output = sum(m["output_tokens"] for m in _metrics_data["calls"].values())

        # per-model breakdown with load distribution and cost estimates
        models = {}
        total_model_calls = 0
        total_paid_input = 0
        total_paid_output = 0
        actual_cost = 0.0

        for name, md in _metrics_data["models"].items():
            total_model_calls += md["count"]

        for name, md in _metrics_data["models"].items():
            pricing = MODEL_PRICING.get(name, {"input": 0.0, "output": 0.0})
            input_cost = (md["input_tokens"] / 1_000_000) * pricing["input"]
            output_cost = (md["output_tokens"] / 1_000_000) * pricing["output"]

            # calculate load %
            load_pct = (
                (md["count"] / total_model_calls * 100) if total_model_calls > 0 else 0
            )

            models[name] = {
                "count": md["count"],
                "input_tokens": md["input_tokens"],
                "output_tokens": md["output_tokens"],
                "total_tokens": md["input_tokens"] + md["output_tokens"],
                "load_pct": round(load_pct, 1),
                "estimated_cost_usd": round(input_cost + output_cost, 6),
            }

            # paid API cost (what it would have been without routing)
            total_paid_input += md["input_tokens"]
            total_paid_output += md["output_tokens"]
            actual_cost += md["estimated_cost"]

        # estimate: if all traffic went to Sonnet at full price
        sonnet_pricing = MODEL_PRICING.get(
            SONNET_PRICING_KEY, _CFG_MODELS["cloud"]["pricing"]
        )
        # Hypothetical: what if every tier query went to Sonnet
        tier_savings = {
            "simple": SONNET_TOKENS_SAVED_PER_SIMPLE,
            "medium": 2500,
            "hard": 4200,
        }
        hypothetical_cost = (
            sum(
                _metrics_data["tier_tokens"][t]["input"]
                + _metrics_data["tier_tokens"][t]["output"]
                for t in ("medium", "hard")
            )
            / 1_000_000
            * sonnet_pricing["input"]
        )
        hypothetical_cost += (
            sum(
                tier_savings[t] * _metrics_data["tiers"].get(t, 0)
                for t in ("simple", "medium", "hard")
            )
            / 1_000_000
            * sonnet_pricing["output"]
        )

        # savings from smart routing
        savings_usd = round(max(0, hypothetical_cost - actual_cost), 6)
        savings_pct = (
            round((savings_usd / hypothetical_cost * 100), 1)
            if hypothetical_cost > 0
            else 0
        )

        # savings score: cumulative badge system
        savings_score = 0
        badge = "bronze"
        if savings_usd >= 100:
            badge = "diamond"
            savings_score = 5
        elif savings_usd >= 50:
            badge = "platinum"
            savings_score = 4
        elif savings_usd >= 20:
            badge = "gold"
            savings_score = 3
        elif savings_usd >= 5:
            badge = "silver"
            savings_score = 2
        elif savings_usd > 0:
            badge = "bronze"
            savings_score = 1

        return JSONResponse(
            content={
                "total_requests": _metrics_data["total_requests"],
                "total_calls": total_calls,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "conversations": _metrics_data["conversations"],
                "errors": _metrics_data["errors"],
                "tiers": dict(_metrics_data["tiers"]),
                "call_breakdown": {
                    k: {
                        "count": v["count"],
                        "input_tokens": v["input_tokens"],
                        "output_tokens": v["output_tokens"],
                    }
                    for k, v in _metrics_data["calls"].items()
                },
                "models": models,
                "cost_summary": {
                    "hypothetical_cost_usd": round(hypothetical_cost, 6),
                    "actual_cost_usd": round(actual_cost, 6),
                    "savings_usd": savings_usd,
                    "savings_pct": savings_pct,
                    "savings_score": savings_score,
                    "badge": badge,
                },
            }
        )


@app.get("/conversations")
async def get_conversations():
    async with _metrics_lock:
        convs = []
        for k, v in _conv_state.items():
            convs.append(
                {
                    "key": k[:80] + "…" if len(k) > 80 else k,
                    "tier": v.get("tier"),
                    "difficulty": v.get("difficulty"),
                }
            )
        return JSONResponse(content={"count": len(convs), "conversations": convs})


@app.get("/dashboard")
async def get_dashboard():
    async with _metrics_lock:
        total_calls = sum(m["count"] for m in _metrics_data["calls"].values())
        total_input = sum(m["input_tokens"] for m in _metrics_data["calls"].values())
        total_output = sum(m["output_tokens"] for m in _metrics_data["calls"].values())

        # per-model breakdown with load distribution and cost estimates
        models = {}
        total_model_calls = 0

        for name, md in _metrics_data["models"].items():
            total_model_calls += md["count"]

        for name, md in _metrics_data["models"].items():
            pricing = MODEL_PRICING.get(name, {"input": 0.0, "output": 0.0})
            input_cost = (md["input_tokens"] / 1_000_000) * pricing["input"]
            output_cost = (md["output_tokens"] / 1_000_000) * pricing["output"]

            load_pct = (
                (md["count"] / total_model_calls * 100) if total_model_calls > 0 else 0
            )

            models[name] = {
                "count": md["count"],
                "input_tokens": md["input_tokens"],
                "output_tokens": md["output_tokens"],
                "total_tokens": md["input_tokens"] + md["output_tokens"],
                "load_pct": round(load_pct, 1),
                "estimated_cost_usd": round(input_cost + output_cost, 6),
            }

        # cost summary
        actual_cost = sum(
            md["estimated_cost"] for md in _metrics_data["models"].values()
        )

        total_paid_input = sum(
            md["input_tokens"] for md in _metrics_data["models"].values()
        )
        total_paid_output = sum(
            md["output_tokens"] for md in _metrics_data["models"].values()
        )

        sonnet_pricing = MODEL_PRICING.get(
            SONNET_PRICING_KEY, _CFG_MODELS["cloud"]["pricing"]
        )
        # Hypothetical: what if every tier query went to Sonnet
        tier_savings = {
            "simple": SONNET_TOKENS_SAVED_PER_SIMPLE,
            "medium": 2500,
            "hard": 4200,
        }
        hypothetical_cost = (
            sum(
                _metrics_data["tier_tokens"][t]["input"]
                + _metrics_data["tier_tokens"][t]["output"]
                for t in ("medium", "hard")
            )
            / 1_000_000
            * sonnet_pricing["input"]
        )
        hypothetical_cost += (
            sum(
                tier_savings[t] * _metrics_data["tiers"].get(t, 0)
                for t in ("simple", "medium", "hard")
            )
            / 1_000_000
            * sonnet_pricing["output"]
        )

        savings_usd = round(max(0, hypothetical_cost - actual_cost), 6)
        savings_pct = (
            round((savings_usd / hypothetical_cost * 100), 1)
            if hypothetical_cost > 0
            else 0
        )

        savings_score_val = 0
        badge = "bronze"
        if savings_usd >= 100:
            badge = "diamond"
            savings_score_val = 5
        elif savings_usd >= 50:
            badge = "platinum"
            savings_score_val = 4
        elif savings_usd >= 20:
            badge = "gold"
            savings_score_val = 3
        elif savings_usd >= 5:
            badge = "silver"
            savings_score_val = 2
        elif savings_usd > 0:
            badge = "bronze"
            savings_score_val = 1

        return JSONResponse(
            content={
                "total_requests": _metrics_data["total_requests"],
                "total_calls": total_calls,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "conversations": len(_conv_state),
                "errors": _metrics_data["errors"],
                "tiers": dict(_metrics_data["tiers"]),
                "call_breakdown": {
                    k: {
                        "count": v["count"],
                        "input_tokens": v["input_tokens"],
                        "output_tokens": v["output_tokens"],
                    }
                    for k, v in _metrics_data["calls"].items()
                },
                "models": models,
                "actual_cost_usd": round(actual_cost, 6),
                "cost_summary": {
                    "hypothetical_cost_usd": round(hypothetical_cost, 6),
                    "actual_cost_usd": round(actual_cost, 6),
                    "savings_usd": savings_usd,
                    "savings_pct": savings_pct,
                    "savings_score": savings_score_val,
                    "badge": badge,
                },
                "recent_activity": _metrics_data["history"][-20:]
                if _metrics_data["history"]
                else [],
                # Full history for cost trend chart (up to 100 points)
                "trend": _metrics_data["history"],
                "active_conversations": [
                    {
                        "key": k[:60] + "…" if len(k) > 60 else k,
                        "tier": v.get("tier"),
                        "difficulty": v.get("difficulty"),
                    }
                    for k, v in list(_conv_state.items())[-20:]
                ],
            }
        )


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/v1/chat/completions")
@app.post("/v1/messages")
async def handle_routing(request: Request):
    raw = await request.body()
    try:
        body_json = json.loads(raw)
    except Exception:
        return Response(content="Invalid JSON", status_code=400)

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    streaming = body_json.get("stream", False)
    messages = body_json.get("messages", [])
    key = conv_key(messages)

    state = _conv_state.get(key, {})
    tier = state.get("tier")
    plan = state.get("plan")
    difficulty = state.get("difficulty")

    # ── Tool loop ──────────────────────────────────────────────────────────────
    if has_tool_turn(messages):
        await record_request(tier or "simple", difficulty or 5)
        log.info("[TOOL TURN] tier=%s", tier)

        exec_msgs = messages
        if tier in ("medium", "hard") and plan:
            exec_msgs = inject_system(
                messages, f"{_WORKER_PREFIX}\n\n[PLAN]\n{plan}\n[/PLAN]"
            )

        err, data = await qwen_execute(exec_msgs, body_json, headers)
        if err:
            return err

        choice = data["choices"][0]
        msg = choice.get("message", {})

        if choice.get("finish_reason") == "tool_calls" or "tool_calls" in msg:
            return (
                make_tool_sse(data)
                if streaming
                else Response(
                    content=json.dumps(data).encode(),
                    status_code=200,
                    media_type="application/json",
                )
            )

        worker_reply = (msg.get("content") or "").strip()
        if tier in ("medium", "hard"):
            polished = await sonnet_polish(
                state.get("question", ""), worker_reply, headers
            )
            final = polished or worker_reply
            plan_used = bool(state.get("plan"))
            reply = build_reply(final, difficulty, tier, plan_used=plan_used)
            model = CLOUD_MODEL if polished else LOCAL_MODEL
            return make_sse(reply, model) if streaming else make_json(data, reply)

        return (
            make_sse(worker_reply, LOCAL_MODEL)
            if streaming
            else make_json(data, worker_reply)
        )

    # ── New user turn — classify ───────────────────────────────────────────────
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    user_question = str(last_user.get("content", "")) if last_user else ""

    if not tier:
        async with _locks_lock:
            if key not in _rating_locks:
                _rating_locks[key] = asyncio.Lock()
        lock = _rating_locks[key]
        async with lock:
            state = _conv_state.get(key, {})
            tier = state.get("tier")
            difficulty = state.get("difficulty")

            if not tier:
                difficulty = (
                    await rate_difficulty(user_question, headers)
                    if user_question
                    else 5
                )
                await record_call("classify", input_text=user_question)
                tier = (
                    "simple"
                    if difficulty <= SIMPLE_MAX
                    else "medium"
                    if difficulty <= MEDIUM_MAX
                    else "hard"
                )
                state = {
                    "tier": tier,
                    "difficulty": difficulty,
                    "question": user_question,
                }
                _conv_state[key] = state

    await record_request(tier, difficulty)
    log.info(
        "[%s] difficulty=%d/10  question=%.60s…",
        tier.upper(),
        difficulty,
        user_question,
    )

    # ── Simple: Qwen3 only ─────────────────────────────────────────────────────
    if tier == "simple":
        err, data = await qwen_execute(messages, body_json, headers)
        if err:
            return err
        choice = data["choices"][0]
        msg = choice.get("message", {})
        if choice.get("finish_reason") == "tool_calls" or "tool_calls" in msg:
            return (
                make_tool_sse(data)
                if streaming
                else Response(
                    content=json.dumps(data).encode(),
                    status_code=200,
                    media_type="application/json",
                )
            )
        reply = build_reply((msg.get("content") or "").strip(), difficulty, "simple")
        return make_sse(reply, LOCAL_MODEL) if streaming else make_json(data, reply)

    # ── Hard (7-10): direct to Sonnet ───────────────────────────────────────
    if tier == "hard":
        log.info("🔴 HARD → direct Sonnet (%d/10)…", difficulty)
        err, data = await _sonnet_direct(messages, body_json, headers)
        if err:
            return err

        choice = data["choices"][0]
        msg = choice.get("message", {})

        if choice.get("finish_reason") == "tool_calls" or "tool_calls" in msg:
            return (
                make_tool_sse(data)
                if streaming
                else Response(
                    content=json.dumps(data).encode(),
                    status_code=200,
                    media_type="application/json",
                )
            )

        reply = build_reply((msg.get("content") or "").strip(), difficulty, "hard")
        return make_sse(reply, CLOUD_MODEL) if streaming else make_json(data, reply)

    # ── Medium (4-6): plan + execute concurrently → polish ───────────────────
    if not plan:
        log.info("🔴 Planning (%s)…", tier)
        # Run plan and execute concurrently; keep both results.
        # No cancellation — whichever finishes last, we use its result.
        plan_task = asyncio.create_task(sonnet_plan(user_question, messages, headers))
        exec_task = asyncio.create_task(qwen_execute(messages, body_json, headers))

        # Wait for BOTH to complete — no cancellation.
        plan_result, (err, data) = await asyncio.gather(
            plan_task, exec_task, return_exceptions=True
        )

        # Handle exceptions from tasks gracefully
        if isinstance(plan_result, Exception):
            log.warning("Plan task failed: %s", plan_result)
            plan_result = None
        if isinstance(err, Exception) or isinstance(data, Exception):
            log.warning("Exec task failed: %s / %s", err, data)
            plan_result = None  # don't store a bad plan
            err, data = (err or data, None)

        if plan_result:
            _conv_state[key]["plan"] = plan_result
            plan = plan_result
    else:
        exec_msgs = inject_system(
            messages, f"{_WORKER_PREFIX}\n\n[PLAN]\n{plan}\n[/PLAN]"
        )
        err, data = await qwen_execute(exec_msgs, body_json, headers)

    if err:
        return err

    if data is None:
        log.error("Both plan and exec failed — no response data")
        return Response(
            content=json.dumps({"error": "All model calls failed"}),
            status_code=502,
        )

    choice = data["choices"][0]
    msg = choice.get("message", {})

    if choice.get("finish_reason") == "tool_calls" or "tool_calls" in msg:
        return (
            make_tool_sse(data)
            if streaming
            else Response(
                content=json.dumps(data).encode(),
                status_code=200,
                media_type="application/json",
            )
        )

    worker_reply = (msg.get("content") or "").strip()

    # ── Polish concurrently with reply assembly to hide latency ────────────
    # Start polish while we build the response metadata; await only if needed.
    polished = None
    if len(worker_reply) >= 200:
        log.info("🔴 Polishing (%d chars)…", len(worker_reply))
        polished_task = asyncio.create_task(
            sonnet_polish(user_question, worker_reply, headers)
        )
        # We need the polish result before returning — await it.
        # For SSE this adds latency but eliminates the sequential bottleneck
        # because plan and exec were already concurrent.
        try:
            polished = await asyncio.wait_for(polished_task, timeout=60.0)
        except asyncio.TimeoutError:
            log.warning("Polish timed out — using Qwen3 answer")
            polished = None
    else:
        log.info("⚡ Skipping polish (short answer: %d chars)", len(worker_reply))

    final = polished or worker_reply
    # plan_used reflects whether Sonnet actually produced a plan for this tier
    plan_used = bool(plan)
    reply = build_reply(final, difficulty, tier, plan_used=plan_used)
    model = CLOUD_MODEL if polished else LOCAL_MODEL
    return make_sse(reply, model) if streaming else make_json(data, reply)


if __name__ == "__main__":
    log.info(
        "Router :9000 | simple(1-%d)→Qwen3 | medium(%d-%d)→Sonnet×2+Qwen3 | hard(7-10)→Sonnet direct",
        SIMPLE_MAX,
        SIMPLE_MAX + 1,
        MEDIUM_MAX,
    )
    uvicorn.run(app, host="127.0.0.1", port=9000)
