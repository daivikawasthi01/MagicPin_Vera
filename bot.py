"""
Vera Bot — Main FastAPI HTTP Server & Challenge API Implementation.
Exposes the 5 challenge endpoints + teardown.
Exports the pure `compose()` library interface.
"""
import os
import time
import asyncio
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from models import (
    HealthzResponse,
    HealthzContextCounts,
    MetadataResponse,
    ContextPushRequest,
    ContextPushResponse,
    TickRequest,
    TickResponse,
    ActionItem,
    ReplyRequest,
    ReplyResponse,
    TeardownResponse
)
from composer import compose, compose_async, call_llm_cascade_async, llm_composition_enabled, get_conversation_id, COMPOSE_CACHE
from reply_fsm import process_reply, GLOBAL_CONVERSATION_STORE, INTENT_QUALIFYING_EXCLUSIONS
from validator import validate_and_clean_action, check_fact_traceability, strip_jargon, strip_urls, JARGON_TERMS
import re

logger = logging.getLogger("vera.bot")
START_TIME = time.time()

# -----------------------------------------------------------------------------
# Global In-Memory RAM Context Store & Suppression Tracking
# -----------------------------------------------------------------------------
contexts: Dict[str, Dict[str, Any]] = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {}
}

context_versions: Dict[str, Dict[str, int]] = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {}
}

_actioned_suppression_keys: set = set()
_pending_suppression_keys: set = set()
_suppression_lock = asyncio.Lock()

# Pre-load seed data from dataset folder as fallback
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
SEED_CATEGORIES: Dict[str, Any] = {}
SEED_MERCHANTS: Dict[str, Any] = {}
SEED_CUSTOMERS: Dict[str, Any] = {}
SEED_TRIGGERS: Dict[str, Any] = {}


def _load_seed_data():
    try:
        cat_dir = DATASET_DIR / "categories"
        if cat_dir.exists():
            for f in cat_dir.glob("*.json"):
                data = json.load(open(f))
                SEED_CATEGORIES[data.get("slug", f.stem)] = data
        for name, container, key in [
            ("merchants_seed.json", SEED_MERCHANTS, "merchant_id"),
            ("customers_seed.json", SEED_CUSTOMERS, "customer_id"),
            ("triggers_seed.json", SEED_TRIGGERS, "id")
        ]:
            path = DATASET_DIR / name
            if path.exists():
                data = json.load(open(path))
                items = data.get(name.split("_")[0], data.get("triggers", data.get("merchants", data.get("customers", []))))
                for item in items:
                    if key in item:
                        container[item[key]] = item
    except Exception as e:
        logger.warning("Seed data preload note: %s", e)


_load_seed_data()


def _trigger_still_active(expires_at: str, now_iso: str) -> bool:
    try:
        exp = datetime.datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        now = datetime.datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
        return exp >= now
    except (ValueError, TypeError):
        return str(expires_at) >= str(now_iso)


app = FastAPI(
    title="Vera Bot",
    description="Context-grounded merchant AI assistant for magicpin AI challenge",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# 0. GET / (Root health check)
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "Vera AI Bot", "healthz": "/v1/healthz"}

# -----------------------------------------------------------------------------
# 1. GET /v1/healthz
# -----------------------------------------------------------------------------
@app.get("/v1/healthz", response_model=HealthzResponse)
async def healthz():
    uptime = int(time.time() - START_TIME)
    counts = HealthzContextCounts(
        category=len(contexts["category"]),
        merchant=len(contexts["merchant"]),
        customer=len(contexts["customer"]),
        trigger=len(contexts["trigger"])
    )
    return HealthzResponse(
        status="ok",
        uptime_seconds=uptime,
        contexts_loaded=counts
    )

# -----------------------------------------------------------------------------
# 2. GET /v1/metadata
# -----------------------------------------------------------------------------
@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata():
    return MetadataResponse(
        bot_name="Vera Bot",
        team_name="Vera AI Team",
        team_members=["Vera Engineer"],
        model="gemini-3.5-flash-lite",
        approach="Deterministic context-grounded trigger dispatch with Gemini Flash 3.5 Lite cascade & validator gates",
        contact_email="vera@example.com",
        version="1.0.0",
        submitted_at="2026-04-26T08:00:00Z",
        author="Vera Team",
        description="Deterministic context-grounded merchant AI engine"
    )

# -----------------------------------------------------------------------------
# 3. POST /v1/context
# -----------------------------------------------------------------------------
@app.post("/v1/context")
async def push_context(req: ContextPushRequest):
    scope = req.scope
    if scope not in contexts:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope '{scope}'"}
        )

    cid = req.context_id
    current_ver = context_versions[scope].get(cid, 0)
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if current_ver > 0 and req.version <= current_ver:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"accepted": False, "reason": "stale_version", "current_version": current_ver}
        )

    contexts[scope][cid] = req.payload
    context_versions[scope][cid] = req.version

    return {
        "accepted": True,
        "ack_id": f"ack_{cid}_v{req.version}",
        "stored_at": now_str
    }

# -----------------------------------------------------------------------------
# 4. POST /v1/tick
# -----------------------------------------------------------------------------
# 4. POST /v1/tick
# -----------------------------------------------------------------------------
@app.post("/v1/tick", response_model=TickResponse)
async def tick(req: TickRequest):
    """
    Periodic wake-up endpoint.
    Filters expired triggers, deduplicates across ticks via suppression keys,
    and produces up to 20 concurrently validated actions.
    """
    now_iso = req.now or "2026-04-26T10:30:00Z"
    raw_triggers = req.available_triggers or []
    
    # 1. Resolve and filter active triggers
    active_triggers: List[Dict[str, Any]] = []
    
    for item in raw_triggers:
        trig_obj = None
        if isinstance(item, str):
            trig_obj = contexts["trigger"].get(item) or SEED_TRIGGERS.get(item)
        elif isinstance(item, dict):
            trig_obj = item
        elif hasattr(item, "model_dump"):
            trig_obj = item.model_dump()

        if not trig_obj or not isinstance(trig_obj, dict):
            continue

        expires_at = trig_obj.get("expires_at")
        if not expires_at or _trigger_still_active(expires_at, now_iso):
            active_triggers.append(trig_obj)

    if not active_triggers:
        return TickResponse(actions=[])

    # 2. Lock-protected deduplication & reservation across ticks
    selected_triggers: List[Dict[str, Any]] = []
    async with _suppression_lock:
        for trig in active_triggers:
            supp_key = trig.get("suppression_key") or trig.get("id") or get_conversation_id(trig)
            if supp_key in _actioned_suppression_keys or supp_key in _pending_suppression_keys:
                continue
            _pending_suppression_keys.add(supp_key)
            selected_triggers.append(trig)
            if len(selected_triggers) >= 20:
                break

    if not selected_triggers:
        return TickResponse(actions=[])

    # 3. Concurrently compose actions for all selected triggers
    async def compose_single(selected_trigger: Dict[str, Any]) -> Optional[ActionItem]:
        supp_key = selected_trigger.get("suppression_key") or selected_trigger.get("id") or get_conversation_id(selected_trigger)
        try:
            m_id = selected_trigger.get("merchant_id")
            if not m_id:
                async with _suppression_lock:
                    _pending_suppression_keys.discard(supp_key)
                return None
            merchant_obj = contexts["merchant"].get(m_id) or SEED_MERCHANTS.get(m_id)
            if not merchant_obj:
                async with _suppression_lock:
                    _pending_suppression_keys.discard(supp_key)
                return None
            cat_slug = merchant_obj.get("category_slug") or selected_trigger.get("payload", {}).get("category")
            if not cat_slug:
                async with _suppression_lock:
                    _pending_suppression_keys.discard(supp_key)
                return None
            category_obj = contexts["category"].get(cat_slug) or SEED_CATEGORIES.get(cat_slug)
            if not category_obj:
                async with _suppression_lock:
                    _pending_suppression_keys.discard(supp_key)
                return None

            customer_id = selected_trigger.get("customer_id")
            customer_obj = (contexts["customer"].get(customer_id) or SEED_CUSTOMERS.get(customer_id)) if customer_id else None

            action_dict = await compose_async(
                category=category_obj,
                merchant=merchant_obj,
                trigger=selected_trigger,
                customer=customer_obj
            )
            if not action_dict or not action_dict.get("body"):
                async with _suppression_lock:
                    _pending_suppression_keys.discard(supp_key)
                return None

            conv_id = action_dict.get("conversation_id")
            payload = selected_trigger.get("payload", {})
            payload = payload if isinstance(payload, dict) else {}
            topic = next((
                payload.get(key) for key in (
                    "intent_topic", "topic", "title", "event", "festival",
                    "service_due", "query", "trend", "theme", "molecule"
                ) if isinstance(payload.get(key), str) and payload.get(key).strip()
            ), None)
            if topic:
                topic = topic.replace("_", " ").strip()
            else:
                kind_label = str(selected_trigger.get("kind") or "current campaign")
                topic = kind_label.replace("_", " ")

            if conv_id:
                GLOBAL_CONVERSATION_STORE.set_active_topic(conv_id, topic)
                GLOBAL_CONVERSATION_STORE.set_active_trigger(conv_id, selected_trigger)
                GLOBAL_CONVERSATION_STORE.record_turn(m_id, conv_id, action_dict.get("body", ""), "vera")

            async with _suppression_lock:
                _actioned_suppression_keys.add(supp_key)
                _pending_suppression_keys.discard(supp_key)

            return ActionItem(**action_dict)
        except Exception as exc:
            logger.warning("Error composing action for trigger %s: %s", selected_trigger.get("id"), exc)
            async with _suppression_lock:
                _pending_suppression_keys.discard(supp_key)
            return None

    action_items = await asyncio.gather(*(
        compose_single(trigger)
        for trigger in selected_triggers
    ))
    valid_actions = [a for a in action_items if a is not None]
    return TickResponse(actions=valid_actions)

# -----------------------------------------------------------------------------
# 5. POST /v1/reply
# -----------------------------------------------------------------------------
@app.post("/v1/reply", response_model=ReplyResponse)
async def reply(req: ReplyRequest):
    """
    Receives replies from simulated merchant or customer.
    Applies Phase C state machine with grounded LLM generation cascade.
    """
    msg_text = ""
    if isinstance(req.message, str):
        msg_text = req.message
    elif isinstance(req.message, dict):
        msg_text = req.message.get("text", "")
    elif hasattr(req.message, "text"):
        msg_text = req.message.text or ""

    merchant_obj = contexts["merchant"].get(req.merchant_id) or SEED_MERCHANTS.get(req.merchant_id)
    cat_slug = (merchant_obj or {}).get("category_slug")
    category_obj = (contexts["category"].get(cat_slug) or SEED_CATEGORIES.get(cat_slug)) if cat_slug else None

    # 1. State machine classification & baseline
    fsm_result = process_reply(
        conversation_id=req.conversation_id,
        merchant_id=req.merchant_id,
        message_text=msg_text,
        turn_number=req.turn_number or 1,
        customer_id=req.customer_id,
        merchant_context=merchant_obj,
        active_topic=GLOBAL_CONVERSATION_STORE.get_active_topic(req.conversation_id)
    )

    action = fsm_result.get("action", "send")
    if action != "send" or not llm_composition_enabled():
        return ReplyResponse(**fsm_result)

    # 2. Grounded LLM reply generation for send actions
    try:
        active_topic = GLOBAL_CONVERSATION_STORE.get_active_topic(req.conversation_id) or "your marketing campaign"
        active_trigger = GLOBAL_CONVERSATION_STORE.get_active_trigger(req.conversation_id)
        owner = (merchant_obj or {}).get("identity", {}).get("owner_first_name", "")
        salutation = f"Dr. {owner}" if owner and cat_slug == "dentists" and not owner.startswith("Dr") else owner
        m_name = (merchant_obj or {}).get("identity", {}).get("name", "your business")

        is_commitment = "confirmed" in fsm_result.get("rationale", "").lower() or "commitment" in fsm_result.get("rationale", "").lower()
        is_qualifying = "qualifying" in fsm_result.get("rationale", "").lower()
        turn_history = GLOBAL_CONVERSATION_STORE.get_turn_history(req.conversation_id)
        history_text = " | ".join([f"{t.get('role')}: {t.get('text')}" for t in turn_history[-3:]]) if turn_history else "None"

        system_prompt = (
            f"You are Vera, an intelligent AI growth partner for Indian local merchants on magicpin.\n"
            f"You are replying on WhatsApp to {salutation or 'the business owner'} of {m_name} ({cat_slug or 'retail'}).\n"
            f"Write a concise (1-2 sentences, max 220 chars), grounded, natural reply.\n"
            f"Never use internal jargon (CTR, algorithm, churn, impressions, triggers).\n"
            f"Never make false claims or ungrounded promises. Avoid emojis."
        )

        if is_commitment:
            user_prompt = (
                f"Topic: {active_topic}\n"
                f"Conversation: {history_text}\n"
                f"Merchant replied: \"{msg_text}\"\n"
                f"Goal: Confirm the draft for {active_topic} is ready and provide the immediate next step to proceed.\n"
                f"Requirements: Use clear action words ('Done', 'Confirmed', 'Here is the draft', 'proceed'). Do NOT ask hesitant qualifying questions."
            )
        elif is_qualifying:
            user_prompt = (
                f"Topic: {active_topic}\n"
                f"Conversation: {history_text}\n"
                f"Merchant question: \"{msg_text}\"\n"
                f"Goal: Answer directly and practically, then ask if they would like to proceed with the draft."
            )
        else:
            user_prompt = (
                f"Topic: {active_topic}\n"
                f"Conversation: {history_text}\n"
                f"Merchant message: \"{msg_text}\"\n"
                f"Goal: Acknowledge and state the next clear step for {active_topic}."
            )

        trigger_id = (active_trigger or {}).get("id", "")
        llm_text = await call_llm_cascade_async(system_prompt, user_prompt, trigger_id)
        if llm_text and len(llm_text.strip()) > 10:
            cleaned = llm_text.strip().strip('"').strip("'")
            if is_commitment:
                lower = cleaned.lower()
                has_qualifying = any(q in lower for q in ["would you", "do you want", "can you tell", "what if", "how about"])
                if not has_qualifying:
                    GLOBAL_CONVERSATION_STORE.record_turn(req.merchant_id, req.conversation_id, cleaned, "vera")
                    return ReplyResponse(
                        action="send",
                        body=cleaned,
                        cta=fsm_result.get("cta", "open_ended"),
                        rationale=fsm_result.get("rationale", "Grounded LLM commitment response.")
                    )
            else:
                GLOBAL_CONVERSATION_STORE.record_turn(req.merchant_id, req.conversation_id, cleaned, "vera")
                return ReplyResponse(
                    action="send",
                    body=cleaned,
                    cta=fsm_result.get("cta", "binary_yes_no"),
                    rationale=fsm_result.get("rationale", "Grounded LLM reply response.")
                )
    except Exception as exc:
        logger.warning("Error in LLM reply generation: %s", exc)

    return ReplyResponse(**fsm_result)

# -----------------------------------------------------------------------------
# 6. POST /v1/teardown
# -----------------------------------------------------------------------------
@app.post("/v1/teardown", response_model=TeardownResponse)
async def teardown():
    """Wipes in-memory contexts, suppression reservations, and conversation history for clean test runs."""
    contexts["category"].clear()
    contexts["merchant"].clear()
    contexts["customer"].clear()
    contexts["trigger"].clear()

    context_versions["category"].clear()
    context_versions["merchant"].clear()
    context_versions["customer"].clear()
    context_versions["trigger"].clear()

    async with _suppression_lock:
        _actioned_suppression_keys.clear()
        _pending_suppression_keys.clear()

    COMPOSE_CACHE.clear()
    GLOBAL_CONVERSATION_STORE.clear()
    return TeardownResponse(status="ok", cleared=True)


__all__ = ["app", "compose", "contexts", "context_versions"]


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)