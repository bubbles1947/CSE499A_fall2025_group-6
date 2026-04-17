"""
RAG router — upload PDFs, manage documents, chat with documents.
"""

import os
import uuid

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.user import User
from app.models.document import Document
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.rag import DocumentResponse, RAGChatRequest, RAGChatResponse
from app.auth import get_current_user
from app.services.rag import process_pdf, query_document

router = APIRouter()
settings = get_settings()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF and ingest it into ChromaDB for RAG."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")

    # Check file size
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Save file to disk
    collection_name = f"doc_{uuid.uuid4().hex[:12]}"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{collection_name}.pdf")
    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record
    doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        chroma_collection=collection_name,
        status="processing",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Process PDF → chunks → embeddings → ChromaDB
    try:
        await process_pdf(file_path, collection_name)
        doc.status = "ready"
    except Exception as e:
        doc.status = "error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}",
        )

    return doc


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents uploaded by the current user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.uploaded_at.desc())
    )
    return result.scalars().all()


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and its ChromaDB collection."""
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Remove ChromaDB collection
    try:
        from app.services.rag import _get_chroma_client
        _get_chroma_client().delete_collection(doc.chroma_collection)
    except Exception:
        pass  # Best-effort cleanup

    # Remove file from disk
    file_path = os.path.join(settings.UPLOAD_DIR, f"{doc.chroma_collection}.pdf")
    if os.path.exists(file_path):
        os.remove(file_path)

    await db.delete(doc)


@router.post("/chat", response_model=RAGChatResponse)
async def rag_chat(
    body: RAGChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Chat with a document using RAG — retrieves relevant chunks then queries the LLM."""
    # Verify document ownership
    result = await db.execute(
        select(Document).where(Document.id == body.document_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.status != "ready":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document is not ready for querying")

    # Get or create conversation
    if body.conversation_id:
        conv_result = await db.execute(
            select(Conversation).where(
                Conversation.id == body.conversation_id,
                Conversation.user_id == current_user.id,
            )
        )
        conversation = conv_result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    else:
        conversation = Conversation(
            user_id=current_user.id,
            model_name=body.model_name,
            title=f"RAG: {doc.filename[:40]}",
            rag_enabled=True,
        )
        db.add(conversation)
        await db.flush()

    # Save user message
    user_msg = Message(conversation_id=conversation.id, role="user", content=body.message)
    db.add(user_msg)
    await db.flush()

    # Query document via RAG pipeline
    answer, sources = await query_document(
        collection_name=doc.chroma_collection,
        query=body.message,
        model_name=body.model_name,
    )

    # Save assistant message
    assistant_msg = Message(conversation_id=conversation.id, role="assistant", content=answer)
    db.add(assistant_msg)
    await db.flush()

    return RAGChatResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation.id,
    )
