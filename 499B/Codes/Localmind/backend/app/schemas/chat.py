"""
Pydantic schemas for chat endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────

class ChatSendRequest(BaseModel):
    conversation_id: Optional[uuid.UUID] = None  # None = create new conversation
    model_name: str = Field(..., description="One of: llama3, mistral, deepseek")
    message: str = Field(..., min_length=1, max_length=10000)
    rag_enabled: bool = False


# ── Responses ─────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    voice_used: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: uuid.UUID
    model_name: str
    title: str
    rag_enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse] = []


class ModelInfo(BaseModel):
    name: str
    display_name: str
    server_url: str
    status: str  # "online" or "offline"
