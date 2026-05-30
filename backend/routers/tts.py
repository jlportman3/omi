"""TTS proxy route — server-side text-to-speech for mobile clients.

Backends selectable via TTS_BACKEND env var:
  - "kokoro"     (default) — local Kokoro-82M via LiteLLM/speaches on rtx6000.
                              No external API costs; voices are Kokoro presets
                              (`af_bella`, `am_michael`, ...). Legacy ElevenLabs
                              voice IDs sent by older clients are translated to
                              their closest Kokoro equivalent.
  - "elevenlabs" (legacy)  — proxies api.elevenlabs.io. Requires
                              ELEVENLABS_API_KEY. Retained for rollback.

Mirrors `desktop/Backend-Rust/src/routes/tts.rs` so mobile clients can play
Omi's spoken responses in background / lock-screen scenarios without shipping
an upstream API key.

Rate limits per user (Redis-backed sliding-window + daily counter):
  - 50 requests per rolling 60 seconds → 429
  - 10,000 characters per UTC day → 429
  - 5,000 characters per single request (hard cap, 400)
"""

import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from database import redis_db
from models.tts import TtsSynthesizeRequest
from utils.http_client import get_tts_client, get_tts_semaphore
from utils.log_sanitizer import sanitize
from utils.other import endpoints as auth

logger = logging.getLogger(__name__)

router = APIRouter()

# Limits mirror desktop/Backend-Rust/src/routes/tts.rs
_TTS_BURST_PER_MINUTE = 50
_TTS_DAILY_CHAR_LIMIT = 10_000
_TTS_BURST_WINDOW_SECS = 60
_TTS_REQUEST_CHAR_LIMIT = 5_000

_ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Which upstream TTS backend to use. "kokoro" (default) routes through the
# local LiteLLM proxy to Kokoro-82M on rtx6000; "elevenlabs" preserves the
# legacy cloud-proxy behavior for rollback.
TTS_BACKEND = os.getenv("TTS_BACKEND", "kokoro").lower()

# Kokoro / LiteLLM configuration. Reuses OPENAI_BASE_URL / OPENAI_API_KEY
# from the existing chat-path setup so we don't introduce new credentials.
_KOKORO_MODEL = os.getenv("TTS_KOKORO_MODEL", "kokoro-tts")
_KOKORO_DEFAULT_VOICE = os.getenv("TTS_KOKORO_DEFAULT_VOICE", "af_bella")
_KOKORO_RESPONSE_FORMAT = os.getenv("TTS_KOKORO_RESPONSE_FORMAT", "mp3")

# Translation table: known ElevenLabs voice IDs that clients still send in
# the field → closest Kokoro voice. Lets old client builds keep working
# after the backend swap without forcing a coordinated app update.
_ELEVENLABS_TO_KOKORO_VOICE = {
    "BAMYoBHLZM7lJgJAmFz0": "af_bella",  # Sloane → Bella (warm American female)
}


def _is_valid_voice_id(voice_id: str, backend: str | None = None) -> bool:
    """Validate a voice ID for the active TTS backend.

    Both backends cap at 128 chars and require a non-empty string. They
    differ on the character class:

    - elevenlabs: alphanumeric only — matches the legacy ID format and
      prevents path traversal against the ElevenLabs URL template
      (e.g. `../../history` would otherwise retarget the xi-api-key).
    - kokoro: alphanumeric + underscore — required to accept the Kokoro
      voice-name convention (e.g. `af_bella`, `am_michael`). Slashes,
      dots, and other path separators are still rejected.
    """
    backend = (backend or TTS_BACKEND).lower()
    if not (1 <= len(voice_id) <= 128):
        return False
    if backend == "kokoro":
        return all(c.isalnum() or c == "_" for c in voice_id)
    return voice_id.isalnum()


def _resolve_voice_for_kokoro(voice_id: str) -> str:
    """Map a legacy ElevenLabs voice ID to its Kokoro equivalent.

    Kokoro-shape voice names pass through unchanged. Unknown legacy IDs also
    pass through — voice-extras will error on them rather than us silently
    substituting a default, so the failure surfaces.
    """
    return _ELEVENLABS_TO_KOKORO_VOICE.get(voice_id, voice_id)


@router.post('/v2/tts/synthesize', tags=['tts'])
async def tts_synthesize(
    req: TtsSynthesizeRequest,
    uid: str = Depends(auth.with_rate_limit(auth.get_current_user_uid, "tts:synthesize")),
):
    """Proxy a TTS request to the active backend (kokoro or elevenlabs).
    Per-user rate limited."""
    if not _is_valid_voice_id(req.voice_id):
        raise HTTPException(status_code=400, detail="invalid voice_id")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    char_count = len(text)
    if char_count > _TTS_REQUEST_CHAR_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds maximum length of {_TTS_REQUEST_CHAR_LIMIT} characters",
        )

    status, retry_after = redis_db.check_tts_rate_limit(
        uid,
        char_count=char_count,
        burst_limit=_TTS_BURST_PER_MINUTE,
        burst_window_secs=_TTS_BURST_WINDOW_SECS,
        daily_char_limit=_TTS_DAILY_CHAR_LIMIT,
    )
    if status == 1:
        logger.warning(f"tts_synthesize: burst rate limit exceeded uid={uid}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded: too many TTS requests. Try again in 60 seconds.",
            headers={"Retry-After": str(retry_after or _TTS_BURST_WINDOW_SECS)},
        )
    if status == 2:
        logger.warning(f"tts_synthesize: daily character limit exceeded uid={uid}")
        raise HTTPException(
            status_code=429,
            detail="Daily TTS character limit exceeded. Resets at midnight UTC.",
            headers={"Retry-After": str(retry_after or 3600)},
        )
    # status == -1 (Redis error): fail-open intentionally — TTS is best-effort.

    if TTS_BACKEND == "kokoro":
        base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not base:
            logger.error("tts_synthesize: OPENAI_BASE_URL not configured for kokoro backend")
            raise HTTPException(status_code=503, detail="TTS service not configured")
        url = f"{base}/audio/speech"
        body = {
            "model": _KOKORO_MODEL,
            "voice": _resolve_voice_for_kokoro(req.voice_id),
            "input": text,
            "response_format": _KOKORO_RESPONSE_FORMAT,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    else:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.error("tts_synthesize: ELEVENLABS_API_KEY not configured")
            raise HTTPException(status_code=503, detail="TTS service not configured")

        body = {
            "text": text,
            "model_id": req.model_id,
            "output_format": req.output_format,
        }
        if req.voice_settings is not None:
            body["voice_settings"] = req.voice_settings.model_dump(exclude_none=True)

        url = _ELEVENLABS_URL.format(voice_id=req.voice_id)
        headers = {
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
            "xi-api-key": api_key,
        }

    client = get_tts_client()
    semaphore = get_tts_semaphore()

    # Acquire the semaphore and open the upstream request OUTSIDE the generator
    # so we can raise a proper HTTPException before StreamingResponse starts
    # writing headers. The generator releases both on exit.
    try:
        await semaphore.acquire()
        try:
            upstream_cm = client.stream("POST", url, json=body, headers=headers, timeout=60.0)
            resp = await upstream_cm.__aenter__()
        except httpx.HTTPError as e:
            semaphore.release()
            logger.error(f"tts_synthesize: upstream request failed uid={uid}: {sanitize(str(e))}")
            raise HTTPException(status_code=502, detail="TTS upstream unavailable")

        if resp.status_code >= 400:
            err_body = await resp.aread()
            err_text = err_body.decode('utf-8', errors='replace')[:200]
            await upstream_cm.__aexit__(None, None, None)
            semaphore.release()
            logger.warning(
                f"tts_synthesize: ElevenLabs returned {resp.status_code} uid={uid}: " f"{sanitize(err_text)}"
            )
            raise HTTPException(status_code=resp.status_code, detail="TTS upstream error")
    except HTTPException:
        raise
    except Exception as e:
        # Defensive: never leak the semaphore on an unexpected failure above.
        try:
            semaphore.release()
        except Exception:
            pass
        logger.error(f"tts_synthesize: pre-stream failure uid={uid}: {sanitize(str(e))}")
        raise HTTPException(status_code=502, detail="TTS upstream unavailable")

    async def audio_stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            try:
                await upstream_cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                semaphore.release()
            except Exception:
                pass

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")
