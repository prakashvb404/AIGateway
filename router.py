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

Tool call design:
  - Tools from the CLIENT REQUEST are passed through transparently to the model.
  - The gateway does NOT execute tools server-side — it returns tool_calls
    back to the client so the client can execute and send tool results back.
  - AVAILABLE_TOOLS in config.json are MERGED with any client-supplied tools,
    allowing the gateway to inject extra tools (e.g. search) on top of whatever
    the client already defines.
"""

import asyncio
import copy
import hashlib
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


def extract_text_content(content: object) -> str:
    """Extract plain text from content (string or array of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    texts.append(item["text"])
                elif "text" in item and not item.get("type"):
                    texts.append(item["text"])
        return "\n".join(texts)
    return ""


def sanitize_for_local(content: object) -> str:
    """Strip MCP/system-reminder noise and return plain string for local model."""
    text = extract_text_content(content)
    cleaned = _SYSTEM_REMINDER_RE.sub("", text).strip()
    return cleaned if cleaned else text


# ── load config ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(_CONFIG_PATH, "r") as f:
        _CFG = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    raise RuntimeError(f"Failed to load config.json: {e}") from e

_CFG_MODELS   = _CFG["models"]
_CFG_ROUTING  = _CFG["routing"]
_CFG_TOKENS   = _CFG["token_budgets"]
_CFG_PROMPTS  = _CFG["prompts"]
_CFG_SERVER   = _CFG["server"]


async def _evict_stale_state():
    """Periodically remove conversation state and locks that haven't been touched in _CONV_TTL."""
    while True:
        await asyncio.sleep(_CONV_TTL)
        now = time.time()
        stale = [k for k, ts in list(_conv_ts.items()) if now - ts > _CONV_TTL]
        if stale:
            for k in stale:
                _conv_state.pop(k, None)
                _conv_ts.pop(k, None)
                _rating_locks.pop(k, None)
            log.info("Evicted %d stale conversation(s)", len(stale))


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_evict_stale_state())
    yield
    task.cancel()
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
    "tier_tokens": {
        "simple": {"input": 0, "output": 0},
        "medium": {"input": 0, "output": 0},
        "hard":   {"input": 0, "output": 0},
    },
    "history": [],
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
    """Rough token estimate with better accuracy for code/JSON."""
    length = len(text)
    if length == 0:
        return 1
    alpha_count  = sum(1 for c in text if c.isalpha())
    alpha_ratio  = alpha_count / length if length > 0 else 0
    if alpha_ratio < 0.5:
        chars_per_token = 2.5
    elif alpha_ratio < 0.7:
        chars_per_token = 3.0
    else:
        chars_per_token = 4.0
    return max(1, int(length / chars_per_token))


async def record_call(call_type: str, input_text: str = "", output_text: str = ""):
    async with _metrics_lock:
        m = _metrics_data["calls"][call_type]
        m["count"] += 1
        if input_text:
            m["input_tokens"]  += _estimate_tokens(input_text)
        if output_text:
            m["output_tokens"] += _estimate_tokens(output_text)


async def record_model_usage(model_name: str, input_tokens: int = 0, output_tokens: int = 0):
    async with _metrics_lock:
        md = _metrics_data["models"][model_name]
        md["count"]         += 1
        md["input_tokens"]  += input_tokens
        md["output_tokens"] += output_tokens
        pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        md["estimated_cost"] = round(
            (md["input_tokens"]  / 1_000_000 * pricing["input"]) +
            (md["output_tokens"] / 1_000_000 * pricing["output"]),
            6,
        )


async def record_request(tier: str, difficulty: int):
    async with _metrics_lock:
        _metrics_data["total_requests"]  += 1
        _metrics_data["conversations"]   += 1
        _metrics_data["tiers"][tier]      = _metrics_data["tiers"].get(tier, 0) + 1
        if tier == "simple":
            _metrics_data["tokens_saved"] += SONNET_TOKENS_SAVED_PER_SIMPLE

        tier_tokens = _metrics_data["tier_tokens"][tier]
        if tier == "simple":
            est_in, est_out = 150, 150
        elif tier == "medium":
            est_in, est_out = 2000, 800
        else:
            est_in, est_out = 3000, 1200
        tier_tokens["input"]  += est_in
        tier_tokens["output"] += est_out

        actual_cost     = sum(md["estimated_cost"] for md in _metrics_data["models"].values())
        sonnet_pricing  = MODEL_PRICING.get(SONNET_PRICING_KEY, _CFG_MODELS["cloud"]["pricing"])
        tier_savings    = {"simple": SONNET_TOKENS_SAVED_PER_SIMPLE, "medium": 2500, "hard": 4200}
        hypothetical    = (
            sum(
                _metrics_data["tier_tokens"][t]["input"] + _metrics_data["tier_tokens"][t]["output"]
                for t in ("medium", "hard")
            ) / 1_000_000 * sonnet_pricing["input"]
        )
        hypothetical += (
            sum(tier_savings[t] * _metrics_data["tiers"].get(t, 0) for t in ("simple", "medium", "hard"))
            / 1_000_000 * sonnet_pricing["output"]
        )
        cumulative_savings = round(max(0, hypothetical - actual_cost), 6)

        _metrics_data["history"].append({
            "time":       time.strftime("%H:%M:%S"),
            "epoch":      time.time(),
            "tier":       tier,
            "difficulty": difficulty,
            "savings":    cumulative_savings,
        })
        if len(_metrics_data["history"]) > 100:
            _metrics_data["history"] = _metrics_data["history"][-100:]


async def record_error():
    async with _metrics_lock:
        _metrics_data["errors"] += 1


# ── endpoints — loaded from config ─────────────────────────────────────────────
LOCAL_AI_URL    = _CFG_MODELS["local"]["url"]
CLOUD_AI_URL    = _CFG_MODELS["cloud"]["url"]
CLASSIFY_AI_URL = _CFG_MODELS["classify"]["url"]

# ── model names — loaded from config ───────────────────────────────────────────
CLOUD_MODEL    = _CFG_MODELS["cloud"]["name"]
LOCAL_MODEL    = _CFG_MODELS["local"]["name"]
CLASSIFY_MODEL = _CFG_MODELS["classify"]["name"]

# ── per-backend auth headers — loaded from config ─────────────────────────────
CLOUD_AUTH    = _CFG_MODELS["cloud"].get("auth", {})
LOCAL_AUTH    = _CFG_MODELS["local"].get("auth", {})
CLASSIFY_AUTH = _CFG_MODELS["classify"].get("auth", {})

# ── token budgets — loaded from config ─────────────────────────────────────────
CLASSIFY_MAX_TOKENS = _CFG_TOKENS["classify_max_tokens"]
PLAN_MAX_TOKENS     = _CFG_TOKENS["plan_max_tokens"]
POLISH_MAX_TOKENS   = _CFG_TOKENS["polish_max_tokens"]
EXECUTE_MAX_TOKENS  = _CFG_TOKENS["execute_max_tokens"]
MAX_TOOL_RESULT_CHARS = _CFG_ROUTING["max_tool_result_chars"]

# ── gateway-level extra tools (injected on top of client tools) ─────────────────
# These are tools the gateway adds to every request (e.g. search, jira).
# The client's own tools are ALWAYS passed through unchanged.
_TOOLS_CONFIG   = _CFG.get("tools", {})
GATEWAY_TOOLS   = _TOOLS_CONFIG.get("definitions", [])   # may be empty []

# ── context window trim — loaded from config ───────────────────────────────────
MAX_CTX_MESSAGES = _CFG_ROUTING["max_ctx_messages"]

# ── difficulty bands — loaded from config ──────────────────────────────────────
SIMPLE_MAX = _CFG_ROUTING["difficulty_bands"]["simple_max"]
MEDIUM_MAX = _CFG_ROUTING["difficulty_bands"]["medium_max"]
HARD_MAX   = 10

# ── prompts — loaded from config ───────────────────────────────────────────────
_CLASSIFY_SYS = _CFG_PROMPTS["classify_sys"]
_PLAN_SYS     = _CFG_PROMPTS["plan_sys"]
_POLISH_SYS   = _CFG_PROMPTS["polish_sys"]
_WORKER_PREFIX = _CFG_PROMPTS["worker_prefix"]

_CONV_TTL = 3600  # seconds before idle conversation state is evicted

# ── state ───────────────────────────────────────────────────────────────────────
_conv_state:    dict[str, dict]          = {}
_conv_ts:       dict[str, float]         = {}  # last-access timestamp per key
_rating_locks:  dict[str, asyncio.Lock]  = {}
_locks_lock = asyncio.Lock()


def _touch_conv(key: str, state: Optional[dict] = None):
    """Update access timestamp and optionally set state for a conversation key."""
    _conv_ts[key] = time.time()
    if state is not None:
        _conv_state[key] = state

# ── shared HTTP client ───────────────────────────────────────────────────────────
_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

_SAFE_HEADERS = {"content-type", "accept", "accept-language", "user-agent"}


def build_backend_headers(req_headers: dict, auth: dict) -> dict:
    """Build headers for a backend call: safe forwarded headers + per-backend auth."""
    h = {k: v for k, v in req_headers.items() if k.lower() in _SAFE_HEADERS}
    h.update(auth)
    return h


# ══════════════════════════════════════════════════════════════════════════════
# Tool helpers
# ══════════════════════════════════════════════════════════════════════════════

def _tool_name(t: dict) -> str:
    """Extract the name from a tool definition, returning empty string if missing."""
    fn = t.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    return (name or t.get("name", "")).strip()


def is_openai_format(tool: dict) -> bool:
    """True if tool uses OpenAI format ({type: 'function', function: {name: ..., parameters: ...}})."""
    return "function" in tool and isinstance(tool["function"], dict)


def is_anthropic_format(tool: dict) -> bool:
    """True if tool uses Anthropic format ({name: ..., input_schema: ..., description: ...})."""
    return "name" in tool and "function" not in tool


def convert_to_anthropic(tool: dict) -> dict:
    """Convert OpenAI-format tool to Anthropic format ({name, description, input_schema})."""
    if is_anthropic_format(tool):
        return tool
    if not is_openai_format(tool):
        return tool
    fn = tool["function"]
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {}),
    }


def convert_to_openai(tool: dict) -> dict:
    """Convert Anthropic-format tool to OpenAI format ({type: 'function', function: {name, description, parameters}})."""
    if is_openai_format(tool):
        return tool
    if not is_anthropic_format(tool):
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def convert_tools_to_anthropic(tools: list) -> list:
    """Convert a list of tools from OpenAI to Anthropic format."""
    return [convert_to_anthropic(t) for t in tools]


def convert_tools_to_openai(tools: list) -> list:
    """Convert a list of tools from Anthropic to OpenAI format."""
    return [convert_to_openai(t) for t in tools]


def merge_tools(client_tools: list) -> list:
    """
    Merge gateway-level extra tools with whatever the client sent.
    Client tools take precedence: if a name clash exists, the client's
    definition wins (we don't override what the caller explicitly set).

    Filters out any tool with an empty or missing name to prevent API errors.
    """
    # Always filter out client tools with empty/whitespace names
    valid_client = [t for t in client_tools if _tool_name(t)]
    if not GATEWAY_TOOLS:
        return valid_client

    client_names = set()
    for t in valid_client:
        client_names.add(_tool_name(t))

    extras = [t for t in GATEWAY_TOOLS
              if _tool_name(t) and _tool_name(t) not in client_names]
    return valid_client + extras


def validate_tools_for_server(tools: list, server_url: str) -> list:
    """Validate tools before sending to a model server.
    
    Filters out tools with empty names and logs warnings for debugging.
    """
    valid = []
    for t in tools:
        name = _tool_name(t)
        if not name:
            log.warning("DROPPING tool with empty name: %s", json.dumps(t)[:200])
            continue
        valid.append(t)
    
    if len(valid) < len(tools):
        log.info("VALIDATED: %d/%d tools passed validation", len(valid), len(tools))
    
    return valid


def extract_tools_from_body(body_json: dict) -> list:
    """Return the tool list the client sent (empty list if none)."""
    return body_json.get("tools", [])


def is_tool_call_response(choice: dict) -> bool:
    """True if this choice contains a tool_calls finish."""
    return (
        choice.get("finish_reason") in ("tool_calls", "tool_use")
        or "tool_calls" in choice.get("message", {})
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def conv_key(messages: list) -> str:
    """Stable conversation fingerprint using the first user message only."""
    first = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    raw = extract_text_content(first) if not isinstance(first, str) else first
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


def trim_messages(messages: list, max_messages: int = MAX_CTX_MESSAGES) -> list:
    """Keep the system prompt + the last `max_messages` non-system messages."""
    system = [m for m in messages if m.get("role") == "system"]
    rest   = [m for m in messages if m.get("role") != "system"]
    return system + rest[-max_messages:]


def strip_tool_messages(messages: list) -> list:
    """Remove tool-call/tool-result messages (used for plan/polish calls)."""
    return [
        m for m in messages
        if m.get("role") not in ("tool",)
        and not (m.get("role") == "assistant" and "tool_calls" in m)
    ]


def inject_system(messages: list, prefix: str) -> list:
    """Replace (or create) the system prompt."""
    messages = list(messages)
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            messages[i] = {**m, "content": prefix}
            return messages
    return [{"role": "system", "content": prefix}] + messages


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


def normalize_tool_results(messages: list) -> list:
    """Convert user messages that look like tool results into proper tool messages.
    
    Some clients send tool results with role='user' instead of role='tool'.
    This detects and fixes that common mistake by checking if a user message
    follows an assistant message containing tool_calls.
    """
    out = []
    prev_was_tool_call = False
    
    for m in messages:
        role = m.get("role", "")
        
        # Check if this is an assistant message with tool_calls
        if role == "assistant" and "tool_calls" in m:
            out.append(m)
            prev_was_tool_call = True
            continue
        
        # If previous was tool_calls and this has a tool_call_id/call_id, convert to tool
        if prev_was_tool_call and role == "user":
            content = m.get("content", "")
            tool_call_id = m.get("tool_call_id") or m.get("call_id") or ""
            if tool_call_id:
                out.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                })
                prev_was_tool_call = False
                continue
            log.warning(
                "DROPPING unidentifiable user message after tool_calls (no tool_call_id). "
                "Client should send role='tool' with tool_call_id."
            )
            prev_was_tool_call = False
            continue
        
        out.append(m)
        prev_was_tool_call = False
    
    return out


def normalize_anthropic_request(body: dict) -> dict:
    """Convert an Anthropic /v1/messages request shape into the internal (OpenAI-ish) shape.
    
    Handles:
    - Top-level `system` field → role: system message
    - Content blocks (text, tool_use, tool_result) → string content / tool_calls / role: tool
    """
    body = {**body}
    messages = list(body.get("messages", []))

    # Top-level system field → prepend a system message
    system = body.pop("system", None)
    if system:
        if isinstance(system, str):
            messages.insert(0, {"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.insert(0, {"role": "system", "content": text})

    # Normalize each message's content blocks
    out = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content")

        if isinstance(content, list):
            # Anthropic content blocks
            texts = []
            tool_calls = []
            for block in content:
                block_type = block.get("type", "")
                if block_type == "text":
                    texts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = " ".join(
                            b.get("text", "") for b in result_content if b.get("type") == "text"
                        )
                    out.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", f"tool_{len(out):03d}"),
                        "content": result_content,
                    })

            merged_text = "\n".join(texts)
            nm = {"role": role}
            if merged_text:
                nm["content"] = merged_text
            else:
                nm["content"] = ""
            if tool_calls:
                nm["tool_calls"] = tool_calls
            out.append(nm)
        else:
            out.append(m)

    body["messages"] = out
    return body


def build_reply(content: str, difficulty: int, tier: str, plan_used: bool = True) -> str:
    local_name = LOCAL_MODEL
    labels = {
        "simple": f"⚡ {local_name} · difficulty {difficulty}/10",
        "medium": f"⚡ {local_name} worker · {'🔴 Sonnet plan+polish' if plan_used else f'{local_name} only'} · difficulty {difficulty}/10",
        "hard":   f"🔴 Sonnet · difficulty {difficulty}/10",
    }
    return f"{content}\n\n---\n*{labels.get(tier, tier)}*"


# ── SSE / response helpers ───────────────────────────────────────────────────────

def _chunk(cid, created, model, delta, finish=None):
    return json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })


def make_sse(content: str, model: str) -> StreamingResponse:
    cid, ts = "chatcmpl-proxy", int(time.time())
    def chunks():
        yield f"data: {_chunk(cid, ts, model, {'role': 'assistant', 'content': ''})}\n\n"
        yield f"data: {_chunk(cid, ts, model, {'content': content})}\n\n"
        yield f"data: {_chunk(cid, ts, model, {}, 'stop')}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(chunks(), media_type="text/event-stream")


def make_tool_sse(data: dict) -> StreamingResponse:
    """Stream a tool_calls response back to the client (OpenAI format)."""
    cid     = data.get("id", "chatcmpl-proxy")
    created = data.get("created", int(time.time()))
    model   = data.get("model", LOCAL_MODEL)
    msg     = data["choices"][0].get("message", {})
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
    return Response(content=json.dumps(out).encode(), status_code=200, media_type="application/json")


def make_tool_json(data: dict) -> Response:
    """Return a tool_calls response as-is (OpenAI format) back to the client."""
    return Response(content=json.dumps(data).encode(), status_code=200, media_type="application/json")


def make_anthropic_json(base: dict, content: str, model: str) -> Response:
    out = {
        "id": base.get("id", f"msg_{int(time.time() * 1000)}"),
        "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": base.get("usage", {"input_tokens": 0, "output_tokens": 0}),
    }
    return Response(content=json.dumps(out).encode(), status_code=200, media_type="application/json")


def make_anthropic_tool_json(data: dict) -> Response:
    """Convert tool call response to Anthropic /v1/messages format."""
    choice  = data["choices"][0]
    msg     = choice.get("message", {})
    content = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []):
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        content.append({
            "type":  "tool_use",
            "id":    tc["id"],
            "name":  tc["function"]["name"],
            "input": args,
        })
    out = {
        "id":            data.get("id", f"msg_{int(time.time() * 1000)}"),
        "type":          "message",
        "role":          "assistant",
        "model":         data.get("model", LOCAL_MODEL),
        "content":       content,
        "stop_reason":   "tool_use" if msg.get("tool_calls") else "end_turn",
        "stop_sequence": None,
        "usage":         data.get("usage", {"input_tokens": 0, "output_tokens": 0}),
    }
    return Response(content=json.dumps(out).encode(), status_code=200, media_type="application/json")


def make_anthropic_sse(content: str, model: str) -> StreamingResponse:
    msg_id = f"msg_{int(time.time() * 1000)}"
    def chunks():
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(content) // 4}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    return StreamingResponse(chunks(), media_type="text/event-stream")


def make_anthropic_tool_sse(data: dict) -> StreamingResponse:
    """Stream a tool_use response back to the client (Anthropic format)."""
    msg_id  = data.get("id", f"msg_{int(time.time() * 1000)}")
    model   = data.get("model", LOCAL_MODEL)
    choice  = data["choices"][0]
    msg     = choice.get("message", {})
    tool_calls = msg.get("tool_calls", [])

    def chunks():
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        for idx, tc in enumerate(tool_calls):
            tool_block = {
                "type": "tool_use",
                "id":   tc["id"],
                "name": tc["function"]["name"],
                "input": {},
            }
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': tool_block})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': tc['function']['arguments']}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    return StreamingResponse(chunks(), media_type="text/event-stream")


# ══════════════════════════════════════════════════════════════════════════════
# Model calls — each sends ONLY what it needs
# ══════════════════════════════════════════════════════════════════════════════

async def rate_difficulty(question: str, headers: dict) -> int:
    """
    Qwen3 classifier — no tools, thinking OFF, 1 message only.
    Returns 1-10 difficulty score.
    """
    q_lower = question.strip().lower()
    if len(q_lower) < 30 and q_lower in {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no", "bye", "goodbye"
    }:
        log.info("🧠 CLASSIFY heuristic: greeting/short → 1")
        return 1

    classify_sys_json = (
        "Rate difficulty 1-10. Return ONLY valid JSON:\n"
        '{"difficulty": <1-10>, "confidence": <0-100>}\n\n'
        "1=greeting/chitchat, 3=simple fact, 5=moderate reasoning, "
        "7=expert knowledge, 10=cutting-edge research."
    )

    async def _classify_once(sys_prompt: str, max_tokens: int) -> Optional[tuple[int, int]]:
        body = {
            "model":    CLASSIFY_MODEL,
            "max_completion_tokens": max_tokens,
            "thinking": {"type": "disabled"},
            # NO tools here — classifier must never see tool definitions
            "messages": [
                {"role": "system",  "content": sys_prompt},
                {"role": "user",    "content": question[:1000]},
            ],
        }
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

            if "usage" in resp:
                usage = resp["usage"]
                await record_model_usage(
                    CLASSIFY_MODEL,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
            else:
                await record_model_usage(
                    CLASSIFY_MODEL,
                    input_tokens=_estimate_tokens(json.dumps(body["messages"])),
                    output_tokens=_estimate_tokens(text),
                )

            try:
                data       = json.loads(text)
                difficulty = max(1, min(10, int(data.get("difficulty", 5))))
                confidence = max(0, min(100, int(data.get("confidence", 0))))
                log.info("🧠 SCORE: %d (confidence: %d%%)", difficulty, confidence)
                return difficulty, confidence
            except (json.JSONDecodeError, KeyError, ValueError):
                log.warning("🧠 CLASSIFY invalid JSON: %r", text)
                return None
        except Exception as e:
            log.warning("CLASSIFY ERROR: %s", e)
            return None

    result = await _classify_once(classify_sys_json, CLASSIFY_MAX_TOKENS)
    if result:
        difficulty, confidence = result
        if confidence >= 60:
            return difficulty
        log.info("🧠 Low confidence (%d%%), defaulting to simple", confidence)
        return 1

    strict_prompt = (
        "Return ONLY this JSON format, nothing else:\n"
        '{"difficulty": 1, "confidence": 90}\n\n'
        "Rate 1-10. 1=greeting, 3=simple, 5=moderate, 7=expert, 10=research."
    )
    result = await _classify_once(strict_prompt, 50)
    if result:
        difficulty, confidence = result
        if confidence >= 60:
            return difficulty

    fallback = 1 if len(question) < 50 else 3
    log.warning("🧠 CLASSIFY failed twice — heuristic fallback: %d", fallback)
    return fallback


async def sonnet_plan(question: str, messages: list, headers: dict) -> Optional[str]:
    """
    Sonnet planner — no tools, stripped history, capped output.
    Only the last few turns + system prompt are sent.
    """
    relevant = strip_tool_messages(messages)
    relevant = trim_messages(relevant, max_messages=4)
    plan_messages = inject_system(relevant, _PLAN_SYS)
    body = {
        "model":  CLOUD_MODEL,
        "stream": False,
        "max_completion_tokens": PLAN_MAX_TOKENS,
        # Planner does NOT need tools — it only produces a textual plan
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
        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        log.info("🔴 PLAN (%d chars): %.200s…", len(plan), plan)
        await record_call("plan", input_text=question, output_text=plan)
        return plan
    except Exception as e:
        log.warning("PLAN ERROR: %s — type: %s", e, type(e).__name__)
        return None


def sanitize_messages(messages: list) -> list:
    """Sanitize content of each message in the list for local model."""
    out = []
    for m in messages:
        content = m.get("content")
        if content is not None:
            m = {**m, "content": sanitize_for_local(content)}
        out.append(m)
    return out


async def qwen_execute(
    messages: list,
    body_json: dict,
    headers: dict,
    client_tools: list,
) -> tuple[Optional[Response], Optional[dict]]:
    """
    Qwen3 worker — thinking ON, trimmed context, transparent tool pass-through.

    client_tools: the tool list from the original client request (may be empty).
    Gateway tools are MERGED in here so the local model sees all of them.
    """
    trimmed = trim_messages(messages, MAX_CTX_MESSAGES)
    trimmed = truncate_tool_results(trimmed)
    trimmed = sanitize_messages(trimmed)
    disable_thinking = should_disable_thinking(trimmed)

    local_body: dict = {
        "model":  LOCAL_MODEL,
        "stream": False,
        "max_completion_tokens": EXECUTE_MAX_TOKENS,
        "messages": trimmed,
        "thinking": {"type": "disabled"} if disable_thinking else {"type": "enabled"},
    }

    # Merge client tools + gateway extras, then attach if non-empty
    all_tools = merge_tools(client_tools)
    # Local model expects OpenAI format; convert from Anthropic if needed
    all_tools = convert_tools_to_openai(all_tools)
    # Validate tools before sending to server (filter empty names)
    all_tools = validate_tools_for_server(all_tools, LOCAL_AI_URL)
    if all_tools:
        local_body["tools"]       = all_tools
        local_body["tool_choice"] = body_json.get("tool_choice", "auto")

    # Debug: log tool count and tool_choice
    log.info("TOOLS=%d TOOL_CHOICE=%s", len(local_body.get("tools", [])), local_body.get("tool_choice"))

    # Pass through safe generation params only
    for key in ("temperature", "top_p", "top_k", "min_p",
                "presence_penalty", "frequency_penalty", "stop"):
        if key in body_json:
            local_body[key] = body_json[key]

    try:
        r = await _client.post(
            LOCAL_AI_URL,
            content=json.dumps(local_body).encode(),
            headers=headers,
            timeout=300.0,
        )
        r.raise_for_status()
        resp = r.json()

        # Promote reasoning_content when content is empty (Qwen3 thinking mode)
        for choice in resp.get("choices", []):
            msg = choice.get("message", {})
            if not msg.get("content") and msg.get("reasoning_content"):
                msg["content"] = msg["reasoning_content"]

        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                LOCAL_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        else:
            await record_model_usage(
                LOCAL_MODEL,
                input_tokens=_estimate_tokens(json.dumps(local_body)),
                output_tokens=_estimate_tokens(json.dumps(resp.get("choices", []))),
            )

        await record_call("execute", input_text=json.dumps(local_body), output_text=json.dumps(resp))
        return None, resp
    except httpx.HTTPStatusError as e:
        log.error("STATUS=%s", e.response.status_code)
        try:
            log.error("BODY=%s", e.response.text)
        except Exception:
            pass
        log.error("REQUEST=%s", json.dumps(local_body, indent=2)[:10000])
        return Response(content=e.response.text, status_code=502), None
    except Exception as e:
        log.error("EXECUTE ERROR: %s (url=%s, model=%s)", e, LOCAL_AI_URL, LOCAL_MODEL)
        return Response(content=json.dumps({"error": str(e)}), status_code=502), None


async def _sonnet_direct(
    messages: list,
    body_json: dict,
    headers: dict,
    client_tools: list,
) -> tuple[Optional[Response], Optional[dict]]:
    """
    Send the full conversation directly to Sonnet (hard tier).
    Client tools are passed through transparently; gateway extras are merged in.
    """
    all_tools = merge_tools(client_tools)
    # Convert to OpenAI format for cloud API
    all_tools = convert_tools_to_openai(all_tools)
    # Validate tools before sending to server (filter empty names)
    all_tools = validate_tools_for_server(all_tools, CLOUD_AI_URL)
    sonnet_body: dict = {
        **body_json,
        "model":    CLOUD_MODEL,
        "stream":   False,
        "messages": messages,
    }
    if all_tools:
        sonnet_body["tools"]       = all_tools
        sonnet_body["tool_choice"] = body_json.get("tool_choice", "auto")
    elif "tools" in sonnet_body:
        # body_json had no tools and we have nothing to add → remove key
        sonnet_body.pop("tools", None)
        sonnet_body.pop("tool_choice", None)

    # Debug: log last 3 messages before model call
    has_tool = any(m.get("role") == "tool" for m in messages)
    log.info(
        "SONNET_DIRECT has_tool_result=%d last_3_roles=[%s]",
        has_tool,
        ", ".join(m.get("role", "?") for m in messages[-3:])
    )

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
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=_estimate_tokens(json.dumps(sonnet_body)),
                output_tokens=_estimate_tokens(json.dumps(resp.get("choices", []))),
            )
        await record_call("direct", input_text=json.dumps(sonnet_body), output_text=json.dumps(resp))
        return None, resp
    except Exception as e:
        log.error("DIRECT SONNET ERROR: %s", e)
        return Response(content=json.dumps({"error": str(e)}), status_code=502), None


async def sonnet_polish(question: str, worker_reply: str, headers: dict) -> Optional[str]:
    """
    Sonnet polisher — question + Qwen3 answer ONLY, no history, no tools.
    """
    body = {
        "model":  CLOUD_MODEL,
        "stream": False,
        "max_completion_tokens": POLISH_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": _POLISH_SYS},
            {"role": "user",   "content": f"Question:\n{question}\n\nAnswer to improve:\n{worker_reply}"},
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
        resp     = r.json()
        polished = resp["choices"][0]["message"]["content"].strip()
        if "usage" in resp:
            usage = resp["usage"]
            await record_model_usage(
                CLOUD_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        log.info("🔴 POLISH (%d chars): %.200s…", len(polished), polished)
        await record_call("polish", input_text=question + worker_reply, output_text=polished)
        return polished
    except Exception as e:
        log.warning("POLISH ERROR: %s — using Qwen3 as-is", e)
        return None


def _return_tool_calls(data: dict, streaming: bool, is_anthropic: bool) -> Response:
    """
    Return a tool_calls response directly to the client WITHOUT executing it.
    The client is responsible for running the tools and sending results back.
    """
    if streaming:
        return make_anthropic_tool_sse(data) if is_anthropic else make_tool_sse(data)
    return make_anthropic_tool_json(data) if is_anthropic else make_tool_json(data)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics / admin endpoints
# ══════════════════════════════════════════════════════════════════════════════

def _compute_cost_summary():
    """Build per-model stats and cost/savings summary from _metrics_data."""
    total_calls  = sum(m["count"]         for m in _metrics_data["calls"].values())
    total_input  = sum(m["input_tokens"]  for m in _metrics_data["calls"].values())
    total_output = sum(m["output_tokens"] for m in _metrics_data["calls"].values())

    models            = {}
    total_model_calls = sum(md["count"] for md in _metrics_data["models"].values())
    actual_cost       = 0.0

    for name, md in _metrics_data["models"].items():
        pricing    = MODEL_PRICING.get(name, {"input": 0.0, "output": 0.0})
        input_cost  = (md["input_tokens"]  / 1_000_000) * pricing["input"]
        output_cost = (md["output_tokens"] / 1_000_000) * pricing["output"]
        load_pct    = (md["count"] / total_model_calls * 100) if total_model_calls > 0 else 0
        models[name] = {
            "count":               md["count"],
            "input_tokens":        md["input_tokens"],
            "output_tokens":       md["output_tokens"],
            "total_tokens":        md["input_tokens"] + md["output_tokens"],
            "load_pct":            round(load_pct, 1),
            "estimated_cost_usd":  round(input_cost + output_cost, 6),
        }
        actual_cost += md["estimated_cost"]

    sonnet_pricing = MODEL_PRICING.get(SONNET_PRICING_KEY, _CFG_MODELS["cloud"]["pricing"])
    tier_savings   = {"simple": SONNET_TOKENS_SAVED_PER_SIMPLE, "medium": 2500, "hard": 4200}
    hypothetical   = (
        sum(
            _metrics_data["tier_tokens"][t]["input"] + _metrics_data["tier_tokens"][t]["output"]
            for t in ("medium", "hard")
        ) / 1_000_000 * sonnet_pricing["input"]
    )
    # hard tier also saves the classify+plan+polish round trips
    hypothetical += (
        sum(tier_savings[t] * _metrics_data["tiers"].get(t, 0) for t in ("simple", "medium", "hard"))
        / 1_000_000 * sonnet_pricing["output"]
    )

    savings_usd = round(max(0, hypothetical - actual_cost), 6)
    savings_pct = round((savings_usd / hypothetical * 100), 1) if hypothetical > 0 else 0

    badge, savings_score = "bronze", 0
    if savings_usd >= 100:    badge, savings_score = "diamond",  5
    elif savings_usd >= 50:   badge, savings_score = "platinum", 4
    elif savings_usd >= 20:   badge, savings_score = "gold",     3
    elif savings_usd >= 5:    badge, savings_score = "silver",   2
    elif savings_usd > 0:     badge, savings_score = "bronze",   1

    return {
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "models": models,
        "cost_summary": {
            "hypothetical_cost_usd": round(hypothetical, 6),
            "actual_cost_usd":       round(actual_cost, 6),
            "savings_usd":           savings_usd,
            "savings_pct":           savings_pct,
            "savings_score":         savings_score,
            "badge":                 badge,
        },
    }


@app.get("/metrics")
async def get_metrics():
    async with _metrics_lock:
        return JSONResponse(content=dict(_metrics_data))


@app.get("/stats")
async def get_stats():
    async with _metrics_lock:
        c = _compute_cost_summary()
        return JSONResponse(content={
            "total_requests":     _metrics_data["total_requests"],
            "conversations":      _metrics_data["conversations"],
            "errors":             _metrics_data["errors"],
            "tiers":              dict(_metrics_data["tiers"]),
            "call_breakdown":     {k: {"count": v["count"], "input_tokens": v["input_tokens"], "output_tokens": v["output_tokens"]} for k, v in _metrics_data["calls"].items()},
            **c,
        })


@app.get("/conversations")
async def get_conversations():
    async with _metrics_lock:
        convs = [
            {"key": k[:80] + "…" if len(k) > 80 else k, "tier": v.get("tier"), "difficulty": v.get("difficulty")}
            for k, v in _conv_state.items()
        ]
        return JSONResponse(content={"count": len(convs), "conversations": convs})


@app.get("/dashboard")
async def get_dashboard():
    async with _metrics_lock:
        c = _compute_cost_summary()
        return JSONResponse(content={
            "total_requests":      _metrics_data["total_requests"],
            "conversations":       len(_conv_state),
            "errors":              _metrics_data["errors"],
            "tiers":               dict(_metrics_data["tiers"]),
            "call_breakdown":      {k: {"count": v["count"], "input_tokens": v["input_tokens"], "output_tokens": v["output_tokens"]} for k, v in _metrics_data["calls"].items()},
            "actual_cost_usd":     c["cost_summary"]["actual_cost_usd"],
            **c,
            "recent_activity":     _metrics_data["history"][-20:] if _metrics_data["history"] else [],
            "trend":               _metrics_data["history"],
            "active_conversations": [
                {"key": k[:60] + "…" if len(k) > 60 else k, "tier": v.get("tier"), "difficulty": v.get("difficulty")}
                for k, v in list(_conv_state.items())[-20:]
            ],
        })


_INDEX_HTML = ""
_INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")
if os.path.exists(_INDEX_PATH):
    with open(_INDEX_PATH, "r") as f:
        _INDEX_HTML = f.read()


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return HTMLResponse(content=_INDEX_HTML)


# ══════════════════════════════════════════════════════════════════════════════
# Main request handler
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
@app.post("/v1/messages")
async def handle_routing(request: Request):
    raw = await request.body()
    try:
        body_json = json.loads(raw)
    except Exception:
        return Response(content="Invalid JSON", status_code=400)

    # Normalize Anthropic /v1/messages requests to internal (OpenAI-ish) shape
    is_anthropic = request.url.path.startswith("/v1/messages")
    if is_anthropic:
        body_json = normalize_anthropic_request(body_json)

    base_headers = {k: v for k, v in request.headers.items() if k.lower() in _SAFE_HEADERS}
    streaming    = body_json.get("stream", False)
    messages     = body_json.get("messages", [])
    
    # Normalize tool results — convert user→tool if client sent wrong schema
    messages = normalize_tool_results(messages)
    
    key          = conv_key(messages)

    # Extract tools the CLIENT sent — these always pass through unchanged
    client_tools = extract_tools_from_body(body_json)

    _touch_conv(key)
    state      = _conv_state.get(key, {})
    tier       = state.get("tier")
    plan       = state.get("plan")
    difficulty = state.get("difficulty")

    # ── Tool result turn (client executed tools and is sending results back) ──
    # Just forward to the right model and return — no classify, no planning.
    if has_tool_turn(messages):
        await record_request(tier or "simple", difficulty or 5)
        log.info("[TOOL RESULT TURN] tier=%s", tier)

        # For medium tier inject the plan so the worker has context
        exec_msgs = messages
        if tier in ("medium", "hard") and plan:
            exec_msgs = inject_system(
                messages, f"{_WORKER_PREFIX}\n\n[PLAN]\n{plan}\n[/PLAN]"
            )

        if tier == "hard":
            err, data = await _sonnet_direct(exec_msgs, body_json, build_backend_headers(base_headers, CLOUD_AUTH), client_tools)
        else:
            err, data = await qwen_execute(exec_msgs, body_json, build_backend_headers(base_headers, LOCAL_AUTH), client_tools)
        if err:
            return err

        choice = data["choices"][0]

        # Model wants to call more tools → return tool_calls to client
        if is_tool_call_response(choice):
            return _return_tool_calls(data, streaming, is_anthropic)

        # Final text answer after tool use
        worker_reply = (choice.get("message", {}).get("content") or "").strip()

        # Polish medium/hard answers
        if tier in ("medium", "hard") and len(worker_reply) >= 200:
            polished = await sonnet_polish(state.get("question", ""), worker_reply, build_backend_headers(base_headers, CLOUD_AUTH))
            final = polished or worker_reply
            model = CLOUD_MODEL if polished else LOCAL_MODEL
        else:
            final = worker_reply
            model = CLOUD_MODEL if tier == "hard" else LOCAL_MODEL

        reply = build_reply(final, difficulty or 5, tier or "simple", plan_used=bool(plan))
        if streaming:
            return make_anthropic_sse(reply, model) if is_anthropic else make_sse(reply, model)
        return make_anthropic_json(data, reply, model) if is_anthropic else make_json(data, reply)

    # ── New user turn — classify ───────────────────────────────────────────────
    last_user     = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    user_question = extract_text_content(last_user.get("content", "")) if last_user else ""
    user_question = sanitize_for_local(user_question)

    if not tier:
        async with _locks_lock:
            if key not in _rating_locks:
                _rating_locks[key] = asyncio.Lock()
        lock = _rating_locks[key]
        async with lock:
            _touch_conv(key)
            state      = _conv_state.get(key, {})
            tier       = state.get("tier")
            difficulty = state.get("difficulty")

            if not tier:
                difficulty = await rate_difficulty(user_question, build_backend_headers(base_headers, CLASSIFY_AUTH)) if user_question else 5
                await record_call("classify", input_text=user_question)
                tier = (
                    "simple" if difficulty <= SIMPLE_MAX else
                    "medium" if difficulty <= MEDIUM_MAX else
                    "hard"
                )
                state = {"tier": tier, "difficulty": difficulty, "question": user_question}
                _touch_conv(key, state)

    await record_request(tier, difficulty)
    log.info("[%s] difficulty=%d/10  question=%.60s…", tier.upper(), difficulty, user_question)

    # ── SIMPLE (1-SIMPLE_MAX): Qwen3 only ─────────────────────────────────────
    if tier == "simple":
        err, data = await qwen_execute(messages, body_json, build_backend_headers(base_headers, LOCAL_AUTH), client_tools)
        if err:
            return err
        choice = data["choices"][0]

        # Model wants to call a tool → return immediately to client
        if is_tool_call_response(choice):
            return _return_tool_calls(data, streaming, is_anthropic)

        reply = build_reply((choice.get("message", {}).get("content") or "").strip(), difficulty, "simple")
        if streaming:
            return make_anthropic_sse(reply, LOCAL_MODEL) if is_anthropic else make_sse(reply, LOCAL_MODEL)
        return make_anthropic_json(data, reply, LOCAL_MODEL) if is_anthropic else make_json(data, reply)

    # ── HARD (7-10): direct to Sonnet ─────────────────────────────────────────
    if tier == "hard":
        log.info("🔴 HARD → direct Sonnet (%d/10)…", difficulty)
        err, data = await _sonnet_direct(messages, body_json, build_backend_headers(base_headers, CLOUD_AUTH), client_tools)
        if err:
            return err
        choice = data["choices"][0]

        if is_tool_call_response(choice):
            return _return_tool_calls(data, streaming, is_anthropic)

        reply = build_reply((choice.get("message", {}).get("content") or "").strip(), difficulty, "hard")
        if streaming:
            return make_anthropic_sse(reply, CLOUD_MODEL) if is_anthropic else make_sse(reply, CLOUD_MODEL)
        return make_anthropic_json(data, reply, CLOUD_MODEL) if is_anthropic else make_json(data, reply)

    # ── MEDIUM (SIMPLE_MAX+1 – MEDIUM_MAX): plan + execute concurrently → polish
    if not plan:
        log.info("🔴 Planning (%s)…", tier)

        plan_task = asyncio.create_task(sonnet_plan(user_question, messages, build_backend_headers(base_headers, CLOUD_AUTH)))
        exec_task = asyncio.create_task(
            qwen_execute(messages, body_json, build_backend_headers(base_headers, LOCAL_AUTH), client_tools)
        )

        plan_result, exec_result = await asyncio.gather(
            plan_task, exec_task, return_exceptions=True
        )

        if isinstance(plan_result, Exception):
            log.warning("Plan task failed: %s", plan_result)
            plan_result = None

        if isinstance(exec_result, Exception):
            log.warning("Exec task failed: %s", exec_result)
            exec_result = (None, None)

        err, data = exec_result  # type: ignore[misc]

        if plan_result:
            state = _conv_state.get(key, {})
            state["plan"] = plan_result
            _touch_conv(key, state)
            plan = plan_result

    else:
        # Subsequent turn in a medium conversation — inject existing plan
        exec_msgs = inject_system(
            messages, f"{_WORKER_PREFIX}\n\n[PLAN]\n{plan}\n[/PLAN]"
        )
        err, data = await qwen_execute(exec_msgs, body_json, build_backend_headers(base_headers, LOCAL_AUTH), client_tools)

    if err:
        return err
    if data is None:
        log.error("Both plan and exec failed — no response data")
        return Response(content=json.dumps({"error": "All model calls failed"}), status_code=502)

    choice = data["choices"][0]

    # Model wants to call a tool → return to client
    if is_tool_call_response(choice):
        return _return_tool_calls(data, streaming, is_anthropic)

    worker_reply = (choice.get("message", {}).get("content") or "").strip()

    # Polish if the answer is substantial
    polished = None
    if len(worker_reply) >= 200:
        log.info("🔴 Polishing (%d chars)…", len(worker_reply))
        try:
            polished = await asyncio.wait_for(
                sonnet_polish(user_question, worker_reply, build_backend_headers(base_headers, CLOUD_AUTH)),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            log.warning("Polish timed out — using Qwen3 answer")
    else:
        log.info("⚡ Skipping polish (short answer: %d chars)", len(worker_reply))

    final     = polished or worker_reply
    plan_used = bool(plan)
    reply     = build_reply(final, difficulty, tier, plan_used=plan_used)
    model     = CLOUD_MODEL if polished else LOCAL_MODEL

    if streaming:
        return make_anthropic_sse(reply, model) if is_anthropic else make_sse(reply, model)
    return make_anthropic_json(data, reply, model) if is_anthropic else make_json(data, reply)


if __name__ == "__main__":
    srv = _CFG_SERVER
    host = srv.get("host", "127.0.0.1")
    port = srv.get("port", 9000)
    log.info(
        "Router %s:%d | simple(1-%d)→Qwen3 | medium(%d-%d)→Sonnet×2+Qwen3 | hard(7-10)→Sonnet direct",
        host, port, SIMPLE_MAX, SIMPLE_MAX + 1, MEDIUM_MAX,
    )
    uvicorn.run(app, host=host, port=port)
