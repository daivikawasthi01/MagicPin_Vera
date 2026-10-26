# Vera Bot

A FastAPI service that composes WhatsApp-style merchant engagement messages for magicpin's **Vera** assistant, built for the magicpin AI Challenge. It turns merchant/customer/trigger context into grounded, on-brand messages, using a deterministic template layer as the safety net and an LLM as an optional upgrade on top.

## Approach

Vera composes each message in two layers. A deterministic template engine (26+ trigger-kind templates) writes a safe, grounded draft first. An LLM (Gemini, with Groq and OpenAI as fallbacks) then gets one shot at rewriting it into something more natural, using a prompt built specifically for that trigger kind. Compliance triggers are told not to sound promotional, planning-intent triggers are told to stick to the merchant's stated topic, and category voice is set explicitly (e.g. "Dr." and clinical register for dentists, energetic tone for gyms). Every LLM output has to pass back through a validator (fact-traceability against the actual context, category taboo words, no jargon leaks, no invented deadlines, no raw ISO dates) before it's allowed to replace the deterministic draft. If it fails, the deterministic version is kept.

Replies are handled separately by a small regex-based state machine (`reply_fsm.py`, exposed for multi-turn via `conversation_handlers.py`) that classifies auto-reply loops, hostile/opt-out messages, and commitment-vs-qualifying language, routing to send/wait/end (no LLM call on this path, since these patterns just need correct routing, not generation). Tick also reserves suppression keys under a lock before composing, to keep concurrent batches from double-composing the same trigger.

## Features

- **Deterministic-first composition** (`composer.py`): 26+ trigger-kind templates with payload-richness fallback, so a response is always produced even if the LLM is unavailable.
- **LLM upgrade pass** with cascading providers: Gemini primary, Groq and OpenAI fallback pools (`gemini_pool.py`, `groq_pool.py`).
- **Category voice profiles** for dentists, salons, restaurants, gyms, and pharmacies.
- **Fact-traceability validator** (`validator.py`): rejects LLM output that invents dates, leaks jargon, uses category taboo words, or can't be traced back to the supplied context.
- **Reply classification state machine** (`reply_fsm.py`, `conversation_handlers.py`): routes merchant replies (auto-reply loops, hostile/opt-out, commitment language) to send/wait/end without an LLM call.
- **FastAPI HTTP API** (`bot.py`) exposing health, metadata, context push, tick (compose), reply, and teardown endpoints.
- **Seed dataset + generator** (`dataset/`): sample categories, merchants, customers, and triggers, with `dataset/generate_dataset.py` to expand them for benchmarking.
- **Benchmarking/evaluation tooling**: `bench.py`, `judge_simulator.py`, `stress_test.py`, and `run_challenge.py` for local scoring and load testing.

## Tech Stack

- **Language**: Python
- **Framework**: FastAPI + Uvicorn
- **Validation**: Pydantic
- **HTTP client**: httpx (LLM calls)
- **LLM providers**: Gemini (primary), Groq, OpenAI (fallback cascade)
- **Config**: python-dotenv
- **Deployment**: Procfile (`uvicorn bot:app`) and `koyeb.yaml` for Koyeb

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file (never commit real keys):

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
LLM_API_KEY=your-key-here
PORT=8080
LOG_LEVEL=INFO
```

### Generate the sample dataset

```bash
python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
```

### Run the server

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
# or, via Procfile:
uvicorn bot:app --host 0.0.0.0 --port ${PORT:-8080}
```

Key endpoints: `GET /v1/healthz`, `GET /v1/metadata`, `POST /v1/context`, `POST /v1/tick`, `POST /v1/reply`, `POST /v1/teardown`.

### Run the full local pipeline (bootstrap + judge)

```bash
python3 run_challenge.py
```

Starts the bot, waits for it to become healthy, then runs the judge simulator against it.

### Benchmark / evaluate

```bash
python3 bench.py          # score compose() outputs against test pairs
python3 judge_simulator.py  # behavioral scenario checks (auto-reply hell, hostile, intent transitions)
python3 stress_test.py      # load testing
```

See `SETUP.md` for infrastructure, hosting, and budget-governance notes for a production deployment, and `challenge-brief.md` / `challenge-testing-brief.md` for the full challenge specification this project was built against.

## Tradeoffs

**Deterministic-first, LLM-as-upgrade**, not the other way around. A judge run under time pressure is exactly the situation where you don't want the median output depending on an API call succeeding, so the deterministic layer alone is built to score solidly and the LLM only ever has upside. Cost: some LLM outputs get reverted for being overly strict about phrasing, leaving some quality on the table in exchange for reliability.

**A tight 4.5s per-call LLM timeout.** Early on, individual calls occasionally took 15-25s under rate limiting, risking the whole tick batch timing out and losing every trigger in it, not just the slow one. Trading a slightly lower LLM-acceptance rate to stop losing entire batches was the right call.

**Per-trigger-kind prompt guidance instead of one shared prompt.** More to maintain, but a generic prompt produced technically-correct-but-flat messages: treating a compliance alert like a marketing trigger, or losing category voice (a yoga studio picking up gym vocabulary). Splitting it out caught failure modes a shared prompt kept reintroducing.

## Additional Context / Future Work

1. **A ground-truth "last customer interaction" field.** Several recall/refill triggers would benefit from a real last-visit or last-refill date instead of inferring or omitting one, since stating a date not in context risks exactly the kind of unverifiable claim fact-traceability is meant to catch.
2. **Merchant-level tone/spice preference.** Energy level is currently inferred from category alone, a decent proxy but not always right within a category (a boutique yoga studio needs a very different register than a general gym).
3. **A canonical address-term taxonomy per customer relationship type** (parent, spouse, self). Built out from observed test cases, but a supplied list would remove guesswork on less common routed-channel scenarios.
