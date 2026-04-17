"""
LocalMind AI — FastAPI application entry point.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables & ensure upload dir exists. Shutdown: cleanup."""
    # Create DB tables (dev convenience — use Alembic migrations in prod)
    await init_db()
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    yield


app = FastAPI(
    title="LocalMind AI",
    description="100% local AI chat platform — LLM inference, RAG, voice, zero cloud dependencies.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React Native dev client and local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ──────────────────────────────────────────────
from app.routers import auth, chat, voice, rag  # noqa: E402

app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(voice.router, prefix="/voice", tags=["Voice"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "LocalMind AI"}
