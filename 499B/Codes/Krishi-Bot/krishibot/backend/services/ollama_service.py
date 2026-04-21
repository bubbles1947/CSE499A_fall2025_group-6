"""
Async client for the local Ollama inference server.

All network I/O is done with httpx AsyncClient. Streaming responses are parsed
line-by-line as NDJSON and tokens are yielded immediately so the router can
push them to the browser via Server-Sent Events or WebSocket without waiting
for the full completion.
"""

import json
import os
from collections.abc import AsyncGenerator

import httpx
from fastapi import HTTPException

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "qwen2.5:3b")

# Shared timeout config — generation can be slow on CPU; 5 min ceiling is safe.
_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)


async def chat_stream(
    messages: list[dict[str, str]],
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    """
    Stream a chat completion token-by-token from Ollama.

    Prepends `system_prompt` as the first message with role ``"system"``
    so the model respects KrishiBot's persona and topic constraints.

    Args:
        messages:      Conversation history as a list of ``{"role": ..., "content": ...}``
                       dicts (user + assistant turns only — no system message).
        system_prompt: The system-role prompt returned by ``prompt_builder``.

    Yields:
        Individual text tokens as they arrive from the model.

    Raises:
        HTTPException 503: If the Ollama server is unreachable.
        HTTPException 502: If Ollama returns a non-200 status.
    """
    payload = {
        "model": TEXT_MODEL,
        "stream": True,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama returned {response.status_code}: {body.decode()}",
                    )

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # Malformed line — skip silently rather than breaking the stream.
                        continue

                    token: str = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token

                    # Ollama sets "done": true on the final chunk.
                    if chunk.get("done"):
                        break

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Ollama. Make sure the server is running on "
            f"{OLLAMA_BASE_URL}.",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out. The model may still be loading.",
        ) from exc


async def analyze_image(base64_image: str, prompt: str) -> str:
    """
    Produce a disease analysis using the text model only.

    The ``base64_image`` parameter is accepted for API compatibility but is
    intentionally NOT sent to Ollama — vision models require significantly more
    memory than the text model, and many users run into "requires more system
    memory" errors. Instead, we rely on filename hints embedded in the prompt
    (see ``prompt_builder.build_image_analysis_prompt``) so the text model can
    produce a plausible diagnosis based on common crop-disease knowledge.

    Args:
        base64_image: Unused — retained for backward-compatible signature.
        prompt:       The analysis prompt built by ``prompt_builder``, which
                      already contains any filename-derived crop hints.

    Returns:
        The model's full response string (expected to be a JSON object,
        possibly wrapped in a markdown code fence — the caller strips those).

    Raises:
        HTTPException 503: If the Ollama server is unreachable.
        HTTPException 502: If Ollama returns a non-200 status.
        HTTPException 504: If generation times out.
    """
    # base64_image is intentionally unused — documented above.
    del base64_image

    payload = {
        "model": TEXT_MODEL,
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned {response.status_code}: {response.text}",
            )

        data = response.json()
        return data.get("message", {}).get("content", "")

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Ollama. Make sure the server is running on "
            f"{OLLAMA_BASE_URL}.",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out. The model may still be loading.",
        ) from exc


async def check_ollama_health() -> bool:
    """
    Probe the Ollama server with a lightweight GET request.

    Uses ``/api/tags`` which returns the list of pulled models — it's fast and
    does not trigger any inference.

    Returns:
        True if the server responds with HTTP 200, False for any other outcome.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except Exception:
        return False
