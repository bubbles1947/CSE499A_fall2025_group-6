"""
Advisory router — returns static crop/topic catalogues and generates
on-demand farming advice by calling the LLM non-streaming.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from schemas.schemas import AdvisoryRequest, AdvisoryResponse
from services import ollama_service, prompt_builder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advisory", tags=["Advisory"])

# ---------------------------------------------------------------------------
# Static catalogues
# ---------------------------------------------------------------------------

_CROPS: list[dict[str, str]] = [
    {"id": "rice",   "name": "Rice",   "name_bn": "ধান",    "icon": "🌾"},
    {"id": "tomato", "name": "Tomato", "name_bn": "টমেটো", "icon": "🍅"},
    {"id": "potato", "name": "Potato", "name_bn": "আলু",   "icon": "🥔"},
    {"id": "wheat",  "name": "Wheat",  "name_bn": "গম",     "icon": "🌿"},
    {"id": "jute",   "name": "Jute",   "name_bn": "পাট",    "icon": "🌱"},
]

_TOPICS: list[dict[str, str]] = [
    {"id": "irrigation",        "name": "Irrigation",         "name_bn": "সেচ"},
    {"id": "fertilizer",        "name": "Fertilizer",         "name_bn": "সার"},
    {"id": "pest_control",      "name": "Pest Control",       "name_bn": "কীটনাশক"},
    {"id": "disease_prevention","name": "Disease Prevention", "name_bn": "রোগ প্রতিরোধ"},
    {"id": "harvest",           "name": "Harvest",            "name_bn": "ফসল কাটা"},
]

_VALID_CROP_IDS  = {c["id"] for c in _CROPS}
_VALID_TOPIC_IDS = {t["id"] for t in _TOPICS}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/crops",
    response_model=list[dict[str, Any]],
    summary="List all supported crops",
)
async def get_crops() -> list[dict[str, str]]:
    """Return the catalogue of crops that KrishiBot can advise on."""
    return _CROPS


@router.get(
    "/topics",
    response_model=list[dict[str, Any]],
    summary="List all advisory topics",
)
async def get_topics() -> list[dict[str, str]]:
    """Return the catalogue of advisory topics available for each crop."""
    return _TOPICS


@router.post(
    "/query",
    response_model=AdvisoryResponse,
    summary="Generate a farming advisory for a specific crop and topic",
)
async def query_advisory(body: AdvisoryRequest) -> AdvisoryResponse:
    """
    Generate a structured advisory by calling the LLM synchronously
    (full response collected before returning — not streamed).

    The crop and topic are validated against the supported catalogues so the
    model always receives well-formed, meaningful inputs.
    """
    # --- Validate crop and topic ---
    if body.crop.lower() not in _VALID_CROP_IDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported crop '{body.crop}'. "
                f"Supported crops: {', '.join(sorted(_VALID_CROP_IDS))}."
            ),
        )
    if body.topic.lower() not in _VALID_TOPIC_IDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported topic '{body.topic}'. "
                f"Supported topics: {', '.join(sorted(_VALID_TOPIC_IDS))}."
            ),
        )

    # --- Build prompt ---
    prompt = prompt_builder.build_advisory_prompt(
        crop=body.crop,
        topic=body.topic,
        question=body.question or "",
    )

    # --- Collect full LLM response (non-streaming) ---
    # We pass the advisory as a single user message with no prior history.
    # The system prompt establishes KrishiBot's persona; the advisory prompt
    # carries all the domain-specific instructions.
    system_prompt = prompt_builder.build_chat_system_prompt("en")
    messages = [{"role": "user", "content": prompt}]

    tokens: list[str] = []
    try:
        async for token in ollama_service.chat_stream(messages, system_prompt):
            tokens.append(token)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error generating advisory for %s/%s", body.crop, body.topic)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the advisory.",
        ) from exc

    content = "".join(tokens).strip()
    if not content:
        raise HTTPException(
            status_code=502,
            detail="The model returned an empty advisory. Please try again.",
        )

    return AdvisoryResponse(crop=body.crop, topic=body.topic, content=content)
