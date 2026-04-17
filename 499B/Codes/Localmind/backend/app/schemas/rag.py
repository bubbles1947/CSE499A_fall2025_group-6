"""
Pydantic schemas for RAG (Retrieval-Augmented Generation) endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────

class RAGChatRequest(BaseModel):
    document_id: uuid.UUID
    model_name: str = Field(..., description="One of: llama3, mistral, deepseek")
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[uuid.UUID] = None


# ── Responses ─────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    chroma_collection: str
    status: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class RAGChatResponse(BaseModel):
    answer: str
    sources: list[str] = []  # Relevant text chunks used for the answer
    conversation_id: uuid.UUID
