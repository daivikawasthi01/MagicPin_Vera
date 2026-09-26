"""
Data structures, schemas, and typed context loaders for Vera Bot.
"""
from typing import Dict, Any, Optional, List, Union, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

# =============================================================================
# API Request / Response Schemas
# =============================================================================

class HealthzContextCounts(BaseModel):
    category: int = 0
    merchant: int = 0
    customer: int = 0
    trigger: int = 0

class HealthzResponse(BaseModel):
    status: str = "ok"
    uptime_seconds: Optional[int] = 0
    contexts_loaded: HealthzContextCounts

class MetadataResponse(BaseModel):
    bot_name: str = "Vera Bot"
    team_name: str = "Vera AI Team"
    team_members: List[str] = Field(default_factory=lambda: ["Vera Engineer"])
    model: str = "llama-3.3-70b-versatile"
    approach: str = "Deterministic context-grounded trigger dispatch with cascade LLM & validator gates"
    contact_email: str = "vera@example.com"
    version: str = "1.0.0"
    submitted_at: str = "2026-04-26T08:00:00Z"
    author: str = "Vera Team"
    description: str = "Deterministic context-grounded merchant AI engine"

class ContextPushRequest(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int = 1
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None

class ContextPushResponse(BaseModel):
    accepted: bool
    ack_id: Optional[str] = None
    stored_at: Optional[str] = None
    reason: Optional[str] = None
    current_version: Optional[int] = None
    details: Optional[str] = None

class TriggerItem(BaseModel):
    id: str
    scope: Optional[Literal["merchant", "customer"]] = "merchant"
    kind: Optional[str] = "generic"
    source: Optional[Literal["external", "internal"]] = "internal"
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)
    urgency: Optional[int] = 1
    suppression_key: Optional[str] = ""
    expires_at: Optional[str] = None

class TickRequest(BaseModel):
    now: Optional[str] = None
    available_triggers: Union[List[TriggerItem], List[str], List[Dict[str, Any]]] = Field(default_factory=list)

class ActionItem(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: Literal["vera", "merchant_on_behalf"] = "vera"
    trigger_id: str
    template_name: str
    template_params: List[Any] = Field(default_factory=list)
    body: str
    cta: Literal["binary_yes_no", "open_ended", "single_choice", "none"] = "open_ended"
    suppression_key: str
    rationale: str

class TickResponse(BaseModel):
    actions: List[ActionItem] = Field(default_factory=list)

class IncomingMessage(BaseModel):
    sender: Optional[str] = "merchant"
    text: Optional[str] = ""

class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    turn_number: Optional[int] = 1
    received_at: Optional[str] = None
    from_role: Optional[str] = "merchant"
    message: Union[IncomingMessage, str, Dict[str, Any]]

class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: Optional[str] = None
    cta: Optional[Literal["binary_yes_no", "open_ended", "single_choice", "none"]] = None
    rationale: Optional[str] = None
    wait_seconds: Optional[int] = None

class TeardownResponse(BaseModel):
    status: str = "ok"
    cleared: bool = True
