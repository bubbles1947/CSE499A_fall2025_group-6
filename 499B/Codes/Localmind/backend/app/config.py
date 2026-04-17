"""
Centralized configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://localmind:localmind_secret@localhost:5432/localmind_db"

    # JWT
    JWT_SECRET_KEY: str = "change-me-to-a-random-secret-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # LLM Servers (llama-cpp-python)
    LLAMA3_SERVER_URL: str = "http://localhost:8001/v1"
    MISTRAL_SERVER_URL: str = "http://localhost:8002/v1"
    DEEPSEEK_SERVER_URL: str = "http://localhost:8003/v1"
    EMBEDDINGS_SERVER_URL: str = "http://localhost:8004/v1"

    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8005

    # Voice
    WHISPER_MODEL_SIZE: str = "base"
    TTS_MODEL_NAME: str = "tts_models/en/ljspeech/tacotron2-DDC"

    # File Uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # silently ignore unknown env vars (e.g. JWT_EXPIRE_DAYS)
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
