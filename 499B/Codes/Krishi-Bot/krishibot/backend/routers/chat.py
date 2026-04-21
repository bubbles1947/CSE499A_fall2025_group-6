"""
Chat router — manages conversation sessions, message persistence, and streaming
LLM responses via Server-Sent Events.
"""

import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models.database import ChatMessage, ChatSession
from schemas.schemas import ChatHistoryResponse, ChatMessageRequest, ChatMessageResponse
from services import ollama_service, prompt_builder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_or_create_session(session_id: str, db: Session) -> ChatSession:
    """Return existing session or create a new one atomically."""
    session = db.query(ChatSession).filter_by(session_id=session_id).first()
    if not session:
        session = ChatSession(session_id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def _get_recent_messages(session_id: str, db: Session, limit: int = 10) -> list[dict]:
    """
    Fetch the last `limit` messages for a session and format them for Ollama.

    Returning the most recent slice keeps the context window small while still
    giving the model enough history to maintain conversation coherence.
    """
    rows = (
        db.query(ChatMessage)
        .filter_by(session_id=session_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(limit)
        .all()
    )
    # Reverse so the messages are chronological (oldest first).
    return [{"role": msg.role, "content": msg.content} for msg in reversed(rows)]


def _save_message(session_id: str, role: str, content: str, db: Session) -> ChatMessage:
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


# ---------------------------------------------------------------------------
# SSE streaming generator
# ---------------------------------------------------------------------------


async def _sse_generator(
    session_id: str,
    user_message: str,
    language: str,
    db: Session,
) -> AsyncGenerator[str, None]:
    """
    Persist the user message, stream the assistant reply token-by-token as SSE,
    then persist the completed assistant message.

    SSE format:   data: <token>\\n\\n
    Terminator:   data: [DONE]\\n\\n
    """
    # 1. Persist user turn before we start streaming.
    _get_or_create_session(session_id, db)
    _save_message(session_id, "user", user_message, db)

    # 2. Build context and system prompt.
    history = _get_recent_messages(session_id, db)
    system_prompt = prompt_builder.build_chat_system_prompt(language)

    # The history already includes the message we just saved; append it as the
    # final user turn so Ollama sees the full conversation in order.
    messages = history  # history ends with the user message we just saved

    # 3. Stream tokens and accumulate the full reply for persistence.
    full_reply: list[str] = []
    try:
        async for token in ollama_service.chat_stream(messages, system_prompt):
            full_reply.append(token)
            # Escape any embedded newlines so they don't break the SSE framing.
            safe_token = token.replace("\n", "\\n")
            yield f"data: {safe_token}\n\n"
    except HTTPException as exc:
        # Surface Ollama errors to the client as a special SSE event.
        yield f"data: [ERROR] {exc.detail}\n\n"
        return
    except Exception as exc:
        logger.exception("Unexpected error during chat stream")
        yield f"data: [ERROR] An unexpected error occurred.\n\n"
        return
    finally:
        # 4. Persist complete assistant reply regardless of how the stream ended.
        if full_reply:
            _save_message(session_id, "assistant", "".join(full_reply), db)

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/message",
    summary="Send a message and stream the assistant reply via SSE",
    response_class=StreamingResponse,
)
async def post_message(
    body: ChatMessageRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Stream the assistant's reply as Server-Sent Events.

    The client should open this with ``EventSource`` or ``fetch`` + ReadableStream
    and collect tokens until it receives ``data: [DONE]``.
    """
    return StreamingResponse(
        _sse_generator(body.session_id, body.message, body.language, db),
        media_type="text/event-stream",
        headers={
            # Prevent nginx / proxies from buffering SSE.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Retrieve the full message history for a session",
)
async def get_history(
    session_id: str,
    db: Session = Depends(get_db),
) -> ChatHistoryResponse:
    session = db.query(ChatSession).filter_by(session_id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    messages = (
        db.query(ChatMessage)
        .filter_by(session_id=session_id)
        .order_by(ChatMessage.timestamp)
        .all()
    )

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessageResponse(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in messages
        ],
    )


@router.delete(
    "/session/{session_id}",
    summary="Delete all messages in a session",
)
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    session = db.query(ChatSession).filter_by(session_id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    db.delete(session)  # cascade deletes all ChatMessage rows via the relationship
    db.commit()
    logger.info("Deleted session %s", session_id)
    return {"message": "Session cleared"}
