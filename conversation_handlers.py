"""
Multi-turn conversation handlers for magicpin AI Challenge.
Implements the optional respond(state, merchant_message) interface for Section 7.4.
"""
from typing import Dict, Any, Optional
from reply_fsm import process_reply, GLOBAL_CONVERSATION_STORE


def respond(state: Any, merchant_message: str) -> Dict[str, Any]:
    """
    Given the conversation so far + the merchant's latest message, produce the reply.
    
    Args:
        state: ConversationState object or dictionary containing context (conversation_id,
               merchant_id, turn_number, active_topic, merchant_context).
        merchant_message: String containing the merchant's latest reply.
        
    Returns:
        Dict with keys: action ('send' | 'wait' | 'end'), body, cta, rationale, etc.
    """
    if isinstance(state, dict):
        conv_id = state.get("conversation_id", "conv_default")
        m_id = state.get("merchant_id", "m_default")
        c_id = state.get("customer_id")
        turn = state.get("turn_number", 1)
        merchant_ctx = state.get("merchant_context")
        topic = state.get("active_topic") or GLOBAL_CONVERSATION_STORE.get_active_topic(conv_id)
    else:
        conv_id = getattr(state, "conversation_id", "conv_default")
        m_id = getattr(state, "merchant_id", "m_default")
        c_id = getattr(state, "customer_id", None)
        turn = getattr(state, "turn_number", 1)
        merchant_ctx = getattr(state, "merchant_context", None)
        topic = getattr(state, "active_topic", None) or GLOBAL_CONVERSATION_STORE.get_active_topic(conv_id)

    return process_reply(
        conversation_id=conv_id,
        merchant_id=m_id,
        message_text=merchant_message,
        turn_number=turn,
        customer_id=c_id,
        merchant_context=merchant_ctx,
        active_topic=topic
    )
