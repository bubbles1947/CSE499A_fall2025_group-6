"""
Chat router — send messages (streaming SSE), list/get/delete conversations, list models.
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.chat import (
    ChatSendRequest,
    ConversationResponse,
    ConversationDetailResponse,
    ModelInfo,
)
from app.auth import get_current_user

router = APIRouter()
settings = get_settings()

# Map model name → llama-cpp-python server URL
MODEL_SERVERS = {
    "llama3": {"display_name": "Llama 3.1 8B", "url": settings.LLAMA3_SERVER_URL},
    "mistral": {"display_name": "Mistral 7B", "url": settings.MISTRAL_SERVER_URL},
    "deepseek": {"display_name": "DeepSeek R1", "url": settings.DEEPSEEK_SERVER_URL},
}


def _get_server_url(model_name: str) -> str:
    """Resolve model name to its llama-cpp-python server URL."""
    info = MODEL_SERVERS.get(model_name)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {model_name}. Choose from: {list(MODEL_SERVERS.keys())}",
        )
    return info["url"]


async def _ping_model_server(server_url: str, model_name: str) -> None:
    """Quick connectivity check. Raises HTTP 503 if the model server is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            resp = await client.get(f"{server_url}/models")
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model_name}' server is not running. "
                f"Start it on {server_url} first."
            ),
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' server returned {e.response.status_code}.",
        )


async def _stream_llm_response(server_url: str, messages: list[dict]) -> AsyncGenerator[str, None]:
    """Stream chat completions from a llama-cpp-python server via SSE."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream(
            "POST",
            f"{server_url}/chat/completions",
            json={
                "messages": messages,
                "stream": True,
                "max_tokens": 2048,
                "temperature": 0.7,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield f"{line}\n\n"
                    if line.strip() == "data: [DONE]":
                        break


@router.post("/send")
async def send_message(
    body: ChatSendRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and stream the LLM response via SSE.
    Creates a new conversation if conversation_id is not provided.
    """
    server_url = _get_server_url(body.model_name)

    # Pre-flight: verify the model server is reachable BEFORE touching the DB.
    # This prevents orphaned conversations/messages when the server is offline.
    await _ping_model_server(server_url, body.model_name)

    # Get or create conversation
    if body.conversation_id:
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == body.conversation_id, Conversation.user_id == current_user.id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    else:
        # Auto-generate title from first message (first 50 chars)
        title = body.message[:50] + ("..." if len(body.message) > 50 else "")
        conversation = Conversation(
            user_id=current_user.id,
            model_name=body.model_name,
            title=title,
            rag_enabled=body.rag_enabled,
        )
        db.add(conversation)
        await db.flush()
        await db.refresh(conversation, attribute_names=["messages"])

    # Build message history for the LLM BEFORE adding user_msg to the session.
    # (Adding user_msg first would auto-append it to conversation.messages via
    #  back_populates, then the manual append below would duplicate it.)
    chat_history = [{"role": m.role, "content": m.content} for m in conversation.messages]
    chat_history.append({"role": "user", "content": body.message})

    # Save user message
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=body.message,
    )
    db.add(user_msg)
    await db.flush()

    # We need to collect the full response to save it to DB
    collected_content = []

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _stream_llm_response(server_url, chat_history):
                data_str = chunk.strip()
                if data_str.startswith("data: ") and data_str != "data: [DONE]":
                    try:
                        payload = json.loads(data_str[6:])
                        delta = payload.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            collected_content.append(delta["content"])
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
                yield chunk
        finally:
            # Save assistant response after streaming completes.
            # get_db() commits the session after the full response is consumed,
            # so a plain add+flush here is enough — no nested transaction needed.
            if collected_content:
                full_response = "".join(collected_content)
                assistant_msg = Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=full_response,
                )
                db.add(assistant_msg)
                await db.flush()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Conversation-Id": str(conversation.id),
        },
    )


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for the current user, newest first."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single conversation with all its messages."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await db.delete(conversation)


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List available LLM models and their real-time health status (concurrent pings)."""
    async def check(name: str, info: dict) -> ModelInfo:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                resp = await client.get(f"{info['url']}/models")
                online = resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            online = False
        return ModelInfo(
            name=name,
            display_name=info["display_name"],
            server_url=info["url"],
            status="online" if online else "offline",
        )

    results = await asyncio.gather(
        *[check(name, info) for name, info in MODEL_SERVERS.items()]
    )
    return list(results)
