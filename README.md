# Vera AI: magicpin AI Challenge Submission

## Approach

Vera composes each message in two layers. A deterministic template engine (26+ trigger-kind templates) writes a safe, grounded draft first. An LLM (Gemini 3.5 Flash Lite, with Groq and OpenAI as fallbacks) then gets one shot at rewriting it into something more natural, using a prompt built specifically for that trigger kind: compliance triggers are told not to sound promotional, planning-intent triggers are told to stick to the merchant's stated topic, category voice is set explicitly (e.g. "Dr." and clinical register for dentists, energetic tone for gyms). Every LLM output has to pass back through a validator (fact-traceability against the actual context, category taboo words, no jargon leaks, no invented deadlines, no raw ISO dates) before it's allowed to replace the deterministic draft. If it fails, we keep the deterministic version.

Replies are handled separately by a small regex-based state machine (`reply_fsm.py`, exposed for multi-turn via `conversation_handlers.py`) that classifies auto-reply loops, hostile/opt-out messages, and commitment-vs-qualifying language, routing to send/wait/end. No LLM call on this path, since these patterns don't need generation, just correct routing. Tick also reserves suppression keys under a lock before composing, to keep concurrent batches from double-composing the same trigger.

## Tradeoffs

**Deterministic-first, LLM-as-upgrade,** not the other way around. A judge run under time pressure is exactly the situation where you don't want the median output depending on an API call succeeding, so the deterministic layer alone is built to score solidly and the LLM only ever has upside. Cost: some LLM outputs get reverted for being overly strict about phrasing, leaving some quality on the table in exchange for reliability.

**A tight 4.5s per-call LLM timeout.** Early on, individual calls occasionally took 15-25s under rate limiting, risking the whole tick batch timing out and losing every trigger in it, not just the slow one. Trading a slightly lower LLM-acceptance rate to stop losing entire batches was the right call.

**Per-trigger-kind prompt guidance instead of one shared prompt.** More to maintain, but a generic prompt produced technically-correct-but-flat messages: treating a compliance alert like a marketing trigger, or losing category voice (a yoga studio picking up gym vocabulary). Splitting it out caught failure modes a shared prompt kept reintroducing.

## Additional context

1. **A ground-truth "last customer interaction" field.** Several recall/refill triggers would benefit from a real last-visit or last-refill date instead of us inferring or omitting one, since stating a date not in context risks exactly the kind of unverifiable claim fact-traceability is meant to catch.
2. **Merchant-level tone/spice preference.** We're inferring energy level from category alone, which is a decent proxy but not always right within a category (a boutique yoga studio needs a very different register than a general gym).
3. **A canonical address-term taxonomy per customer relationship type** (parent, spouse, self). Built this out from observed test cases, but a supplied list would remove guesswork on the less common routed-channel scenarios.
