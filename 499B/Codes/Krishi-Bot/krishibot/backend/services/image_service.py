"""
Image preprocessing utilities for KrishiBot.

Pillow is used for resizing and format normalisation. All functions operate on
raw bytes so they remain transport-agnostic (works with FastAPI UploadFile,
base64-decoded data, or anything else).
"""

import base64
import io

from PIL import Image, UnidentifiedImageError


def preprocess_image(file_bytes: bytes, max_size: int = 1024) -> bytes:
    """
    Resize and normalise an image for vision-model inference.

    Steps:
    1. Open the image from raw bytes.
    2. Convert to RGB — strips alpha channel from PNGs and handles palette modes.
    3. Downscale so the longest side is at most `max_size` pixels while
       preserving the original aspect ratio. Images already within the limit
       are returned unchanged (no upscaling).
    4. Re-encode as JPEG at quality=85 to keep the payload small.

    Args:
        file_bytes: Raw image bytes (JPEG, PNG, WebP, etc.).
        max_size:   Maximum pixels on the longest side (default 1024).

    Returns:
        JPEG-encoded bytes of the processed image.

    Raises:
        ValueError: If `file_bytes` cannot be decoded as a valid image.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Cannot decode image — unsupported or corrupted file.") from exc

    # Normalise colour mode; RGBA/P/L → RGB so JPEG encoding never fails.
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    longest = max(width, height)

    if longest > max_size:
        scale = max_size / longest
        new_size = (int(width * scale), int(height * scale))
        # LANCZOS gives the best quality for downscaling.
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def image_to_base64(file_bytes: bytes) -> str:
    """
    Encode raw image bytes as a base64 string.

    Returns a plain base64 string with no data-URI prefix (``data:image/...``).
    The Ollama API expects the bare string; callers that need a data URI can
    prepend the prefix themselves.

    Args:
        file_bytes: Raw image bytes.

    Returns:
        Base64-encoded string.
    """
    return base64.b64encode(file_bytes).decode("utf-8")


def validate_image(file_bytes: bytes) -> bool:
    """
    Check whether `file_bytes` represent a valid, decodable image.

    Uses Pillow's ``verify()`` which performs a lightweight structural check
    without fully decoding pixel data.

    Args:
        file_bytes: Raw bytes to validate.

    Returns:
        True if the bytes form a valid image, False otherwise.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        return True
    except Exception:
        return False
