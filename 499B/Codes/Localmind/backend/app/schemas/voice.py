"""
Pydantic schemas for voice endpoints.
"""

from pydantic import BaseModel


# ── Responses ─────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    text: str
    language: str


class SynthesisResponse(BaseModel):
    """Response metadata — actual audio is returned as file download."""
    message: str = "Audio synthesized successfully"
    duration_seconds: float


# ── Requests ──────────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str
