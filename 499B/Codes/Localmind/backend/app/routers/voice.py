"""
Voice router — local Whisper STT and Coqui TTS.
"""

import io
import os
import shutil
import tempfile
import time

# Ensure ffmpeg is findable — winget installs it here on Windows
_FFMPEG_BIN = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft", "WinGet", "Packages",
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "ffmpeg-8.1-full_build", "bin",
)
if _FFMPEG_BIN and os.path.isdir(_FFMPEG_BIN) and _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
    print(f"[Voice] Added ffmpeg to PATH: {_FFMPEG_BIN}")

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.auth import get_current_user
from app.models.user import User
from app.schemas.voice import TranscriptionResponse, SynthesizeRequest

router = APIRouter()

# Lazy-loaded singletons to avoid slow import on every request
_whisper_model = None
_tts_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        from app.config import get_settings
        _whisper_model = whisper.load_model(get_settings().WHISPER_MODEL_SIZE)
    return _whisper_model


def _get_tts():
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        from app.config import get_settings
        _tts_model = TTS(model_name=get_settings().TTS_MODEL_NAME)
    return _tts_model


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, m4a, webm, ogg)"),
    current_user: User = Depends(get_current_user),
):
    """Transcribe audio to text using local Whisper model."""
    ct = (audio.content_type or "").lower()
    fn = audio.filename or "audio.webm"
    print(f"[TRANSCRIBE] file={fn!r}  content-type={ct!r}")

    # Reject only obvious non-audio types — use startswith to handle
    # codec suffixes like "audio/webm;codecs=opus" from browser MediaRecorder
    if ct and not ct.startswith("audio/") and not ct.startswith("video/webm"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ct}",
        )

    # Determine file extension: prefer filename, fall back to content-type
    suffix = os.path.splitext(fn)[1]
    if not suffix:
        if "webm" in ct or "webm" in fn:
            suffix = ".webm"
        elif "mp4" in ct or "m4a" in ct:
            suffix = ".m4a"
        elif "mpeg" in ct or "mp3" in fn:
            suffix = ".mp3"
        else:
            suffix = ".wav"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await audio.read()
        print(f"[TRANSCRIBE] received {len(content)} bytes, suffix={suffix!r}")
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)

        model = _get_whisper()
        result = model.transcribe(tmp_path)
        text = result["text"].strip()
        print(f"[TRANSCRIBE] result: {text!r}")
        return TranscriptionResponse(
            text=text,
            language=result.get("language", "en"),
        )
    except Exception as exc:
        print(f"[TRANSCRIBE] ERROR: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/synthesize")
async def synthesize(
    body: SynthesizeRequest,
    current_user: User = Depends(get_current_user),
):
    """Convert text to speech using local Coqui TTS. Returns WAV audio."""
    return await _run_tts(body.text)


@router.get("/synthesize_get")
async def synthesize_get(
    text: str,
    current_user: User = Depends(get_current_user),
):
    """GET variant of /synthesize — used by React Native FileSystem.downloadAsync."""
    return await _run_tts(text)


async def _run_tts(text: str):
    if not text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text cannot be empty")

    tts = _get_tts()
    start = time.time()

    # Use a named temp path — TTS holds the file handle internally, so we
    # read it into memory before unlinking (Windows WinError 32 workaround).
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)  # Close our FD; TTS will open it itself

    try:
        tts.tts_to_file(text=text, file_path=tmp_path)
        duration = time.time() - start

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # Already deleted or still locked — non-fatal

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav",
            "X-Duration-Seconds": f"{duration:.2f}",
        },
    )
