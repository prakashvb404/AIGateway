# AI Router

A token-disciplined multi-model AI request router that intelligently distributes LLM requests across cloud and local models based on question difficulty — saving cost while maintaining quality.

## How It Works

AI Router sits between your application and AI model APIs as a proxy. Every incoming question is rated for difficulty (1–10) and routed through an optimal path:

| Tier | Difficulty | Models Used | Strategy |
|------|-----------|-------------|----------|
| **Simple** | 1–3 | Qwen3 (local) only | Direct execution, zero cloud cost |
| **Medium** | 4–7 | Sonnet + Qwen3 | Concurrent plan → execute → polish |
| **Hard** | 8–10 | Sonnet + Qwen3 | Same as medium |

### The Three-Model Pipeline

```
User Question
    │
    ▼
┌──────────────┐
│ Classify     │  GPT-4o-mini, 10 tokens ──► difficulty score 1–10
└──────┬───────┘
       │
       ├─ Simple (1–3) ──────────────────────────► Qwen3 local: direct answer
       │
       ├─ Medium (4–7) ──────────────────────────► Sonnet plan + Qwen3 execute (concurrent)
       │                                            └─► Sonnet polish
       │
       └─ Hard (8–10) ───────────────────────────► Sonnet plan + Qwen3 execute (concurrent)
                                                  └─► Sonnet polish
```

Every model call sends only what that specific call needs — strict token caps prevent waste.

## Features

- **Token discipline** — each model call has a hard token budget (classify: 10, plan: 400, polish: 500, execute: 8192)
- **Concurrent execution** — planning and execution run in parallel for medium/hard questions
- **Cost tracking** — real-time per-model cost estimation with cumulative savings dashboard
- **Conversation state** — maintains context across turns with intelligent trimming (max 12 messages)
- **Tool support** — handles tool-calling loops between local model and external tools
- **Live dashboard** — dark-themed, auto-refreshing metrics UI with tier distribution and savings tracking
- **OpenAI-compatible API** — exposes `/v1/chat/completions` and `/v1/messages` endpoints

## Project Structure

```
ai-router/
├── router.py          # FastAPI application — routing logic, API endpoints, metrics
├── config.json        # Model URLs, pricing, token budgets, prompts, server settings
├── index.html         # Dashboard UI (served at /)
└── frontend/          # Additional static assets
```

## Getting Started

### Prerequisites

- Python 3.10+
- A local LLM server (e.g., llama.cpp serving Qwen3 on port 8081)
- An OpenAI-compatible API for the classify and polish models

### Installation

```bash
git clone <repository-url>
cd ai-router
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Configuration

Edit `config.json` to set your model endpoints, pricing, and behavior:

```json
{
  "models": {
    "cloud": {
      "name": "claude-sonnet-4.6",
      "url": "http://localhost:4141/v1/chat/completions",
      "pricing": { "input": 3.0, "output": 15.0 }
    },
    "local": {
      "name": "Qwen3-35B",
      "url": "http://localhost:8081/v1/chat/completions",
      "pricing": { "input": 0.0, "output": 0.0 }
    },
    "classify": {
      "name": "gpt-4o-mini",
      "url": "http://localhost:4141/v1/chat/completions",
      "pricing": { "input": 0.1, "output": 0.4 }
    }
  },
  "routing": {
    "difficulty_bands": { "simple_max": 3, "medium_max": 7 },
    "max_ctx_messages": 12,
    "max_tool_result_chars": 8000
  },
  "token_budgets": {
    "classify_max_tokens": 10,
    "plan_max_tokens": 400,
    "polish_max_tokens": 500,
    "execute_max_tokens": 8192
  },
  "prompts": {
    "classify_sys": "Rate difficulty for a local model. Reply with ONE digit 1-10...",
    "plan_sys": "You are an orchestrator... Write a numbered execution plan only...",
    "polish_sys": "Senior reviewer. Fix errors, fill gaps, improve clarity...",
    "worker_prefix": "You are a worker agent. Follow the plan exactly..."
  },
  "server": { "host": "127.0.0.1", "port": 9000 }
}
```

### Running

```bash
python router.py
```

The server starts on `http://127.0.0.1:9000` by default.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (main proxy) |
| `/v1/messages` | POST | Anthropic-compatible messages API |
| `/` | GET | Live metrics dashboard |
| `/metrics` | GET | Raw metrics JSON |
| `/stats` | GET | Aggregated stats with cost summary and savings badges |
| `/conversations` | GET | List of active conversations with tier/difficulty data |
| `/dashboard` | GET | Full dashboard data including recent activity and trends |

### Example: Chat Completion

```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

## Dashboard

Visit `http://127.0.0.1:9000/` for a live dashboard showing:

- **Savings hero banner** — total tokens saved and cost avoided by routing simple questions to local model
- **Tier distribution** — pie chart of requests across simple/medium/hard tiers
- **Model usage cards** — per-model token counts, call counts, and estimated costs
- **Recent activity table** — last 20 requests with tier, difficulty, response time, and cost
- **Savings trend chart** — rolling window showing cumulative tokens saved over time

Dashboard auto-refreshes every 5 seconds.

## Architecture

### Key Components

- **Difficulty Classifier** (`rate_difficulty`) — rates each question 1–10 using the classify model with a 10-token budget
- **Conversation State Manager** (`_conv_state`) — tracks tier, difficulty, plan, and question per conversation via HMAC-SHA256 keys
- **Message Processor** — trims context to max messages, strips tool messages for Sonnet calls, truncates oversized tool results
- **Concurrent Runner** — uses `asyncio.gather` to run planning and execution in parallel for medium/hard questions
- **Metrics Tracker** — per-model token counting with cost estimation from configurable pricing data

### Token Budgets

| Call Type | Model | Max Output Tokens | Purpose |
|-----------|-------|-------------------|---------|
| `classify` | Classify model | 10 | Rate question difficulty |
| `plan` | Sonnet (cloud) | 400 | Generate numbered execution steps |
| `polish` | Sonnet (cloud) | 500 | Improve local model's answer |
| `execute` | Qwen3 (local) | 8192 | Execute plan and produce final answer |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
