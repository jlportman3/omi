import io
import os
import wave
from collections import defaultdict
from io import BytesIO
from typing import List, Optional, Sequence, Tuple, Union

import fal_client
import httpx
from deepgram import DeepgramClient, DeepgramClientOptions

from models.transcript_segment import TranscriptSegment
from utils.byok import get_byok_key
from utils.other.endpoints import timeit
import logging

_DG_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

logger = logging.getLogger(__name__)

# Initialize Deepgram client for pre-recorded transcription
# WARN: the pre-recorded transcription is available on deepgram cloud
_deepgram_options = DeepgramClientOptions(options={"keepalive": "true"})
_deepgram_client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'), _deepgram_options)

# Which batch STT backend to use. "whisper" (default) routes through the local
# LiteLLM proxy to faster-whisper-large-v3 on rtx6000, with optional Sortformer
# diarization via voice-extras. "deepgram" preserves the legacy cloud path for
# rollback (set STT_BATCH_BACKEND=deepgram + DEEPGRAM_API_KEY).
STT_BATCH_BACKEND = os.getenv("STT_BATCH_BACKEND", "whisper").lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
WHISPER_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://10.0.60.48:4000/v1").rstrip("/")
VOICE_EXTRAS_URL = os.getenv("VOICE_EXTRAS_URL", "http://10.0.60.48:8094").rstrip("/")
SORTFORMER_MODEL = os.getenv("SORTFORMER_MODEL", "sortformer-stream")


def _deepgram_client_for_request() -> DeepgramClient:
    """Route to BYOK Deepgram key when set; otherwise use the process-wide client."""
    byok = get_byok_key('deepgram')
    if byok:
        return DeepgramClient(byok, _deepgram_options)
    return _deepgram_client


# Languages supported by nova-3
_deepgram_nova3_languages = {
    "ar",
    "ar-AE",
    "ar-SA",
    "ar-QA",
    "ar-KW",
    "ar-SY",
    "ar-LB",
    "ar-PS",
    "ar-JO",
    "ar-EG",
    "ar-SD",
    "ar-TD",
    "ar-MA",
    "ar-DZ",
    "ar-TN",
    "ar-IQ",
    "ar-IR",
    "be",
    "bg",
    "bn",
    "bs",
    "ca",
    "cs",
    "da",
    "da-DK",
    "de",
    "de-CH",
    "el",
    "en",
    "en-US",
    "en-AU",
    "en-GB",
    "en-IN",
    "en-NZ",
    "es",
    "es-419",
    "et",
    "fa",
    "fi",
    "fr",
    "fr-CA",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "kn",
    "ko",
    "ko-KR",
    "lt",
    "lv",
    "mk",
    "mr",
    "ms",
    "nl",
    "nl-BE",
    "no",
    "pl",
    "pt",
    "pt-BR",
    "pt-PT",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "sv-SE",
    "ta",
    "te",
    "th",
    "th-TH",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
    "zh-CN",
    "zh-Hans",
    "zh-HK",
    "zh-Hant",
    "zh-TW",
}


def get_deepgram_model_for_language(language: str) -> Tuple[str, str]:
    """
    Determine the appropriate Deepgram model and language for pre-recorded transcription.

    Args:
        language: The requested language code or 'multi' for auto-detection

    Returns:
        Tuple of (language_to_use, model_name)
    """
    # For multi-language mode
    if language == 'multi':
        return 'multi', 'nova-3'

    # Languages supported by nova-3
    if language in _deepgram_nova3_languages:
        return language, 'nova-3'

    # Unsupported language - fall back to multi for auto-detection
    return 'multi', 'nova-3'


def _wav_wrap_pcm(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit signed PCM in a WAV header.

    The local Whisper endpoint expects a decodable audio file. Callers that
    pass raw PCM (encoding='linear16') must have their bytes wrapped before
    upload; this helper produces a minimal RIFF/WAVE container in memory.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _merge_words_with_speakers(whisper_words: List[dict], diarization_segments: List[dict]) -> List[dict]:
    """Merge per-word Whisper timestamps with Sortformer speaker segments.

    Returns the legacy Deepgram-compatible word-dict shape:
        {'timestamp': [start, end], 'speaker': 'SPEAKER_XX', 'text': 'word'}

    Each Whisper word is assigned a speaker by looking up the diarization
    segment whose time-range contains the word's midpoint. Words that fall
    outside any segment (gaps in Sortformer coverage) default to SPEAKER_00.
    When no diarization segments are present at all, every word is SPEAKER_00.
    """
    out: List[dict] = []
    for w in whisper_words:
        start = float(w.get("start", 0.0))
        end = float(w.get("end", 0.0))
        midpoint = (start + end) / 2
        speaker = "SPEAKER_00"
        for s in diarization_segments:
            try:
                if s["start"] <= midpoint <= s["end"]:
                    idx = int(str(s.get("speaker", "speaker_0")).split("_")[-1])
                    speaker = f"SPEAKER_{idx:02d}"
                    break
            except (KeyError, ValueError, TypeError):
                continue
        text = (w.get("word") or "").strip()
        out.append({"timestamp": [start, end], "speaker": speaker, "text": text})
    return out


def _post_whisper(audio_bytes: bytes, language: Optional[str], model: Optional[str]) -> dict:
    """POST audio to LiteLLM /v1/audio/transcriptions and return the parsed JSON."""
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {
        "model": model or WHISPER_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    if language and language != "multi":
        data["language"] = language
    headers = {}
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = httpx.post(
        f"{WHISPER_BASE_URL}/audio/transcriptions",
        files=files,
        data=data,
        headers=headers,
        timeout=_DG_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _post_sortformer(audio_bytes: bytes) -> List[dict]:
    """POST audio to voice-extras /v1/audio/diarization and return the segments list.

    Falls back to an empty list if Sortformer fails — the caller will then
    default every word to SPEAKER_00, which is better than failing the whole
    transcription over a diarization hiccup.
    """
    try:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": SORTFORMER_MODEL}
        resp = httpx.post(
            f"{VOICE_EXTRAS_URL}/v1/audio/diarization",
            files=files,
            data=data,
            timeout=_DG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("segments") or []
    except Exception as e:
        logger.warning(f"Sortformer diarization failed, falling back to single-speaker: {e}")
        return []


@timeit
def local_whisper_prerecorded_from_bytes(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    diarize: bool = True,
    attempts: int = 0,
    encoding: Optional[str] = None,
    channels: int = 1,
    language: Optional[str] = None,
    model: Optional[str] = None,
    return_language: bool = False,
    keywords: Optional[Sequence[str]] = None,
) -> Union[List[dict], Tuple[List[dict], str]]:
    """Local Whisper (+optional Sortformer) replacement for deepgram_prerecorded_from_bytes.

    Produces the same legacy word-dict shape so callers do not need to change.
    Raw PCM (encoding='linear16') is wrapped in a WAV header before upload.

    Args:
        audio_bytes: WAV file bytes or raw PCM (set encoding='linear16').
        sample_rate: Sample rate of the source. Required for raw PCM.
        diarize: When True, call Sortformer and assign per-word speakers.
        attempts: Retry counter (currently single-attempt; reserved for parity).
        encoding: 'linear16' for raw PCM, None for WAV.
        channels: Mono / stereo channel count of the source PCM.
        language: ISO code or 'multi' for auto-detect.
        model: Override WHISPER_MODEL alias if needed.
        return_language: When True, returns (words, detected_language).
        keywords: Accepted but unused — Whisper does not support keyterm boost.
    """
    logger.info(
        f"local_whisper_prerecorded_from_bytes bytes_len={len(audio_bytes)} "
        f"encoding={encoding} sample_rate={sample_rate} diarize={diarize} language={language}"
    )

    if encoding == "linear16":
        audio_bytes = _wav_wrap_pcm(audio_bytes, sample_rate=sample_rate, channels=channels)

    payload = _post_whisper(audio_bytes, language=language, model=model)
    words = payload.get("words") or []
    detected_lang = (payload.get("language") or "en").split("-")[0]

    if not words:
        if return_language:
            return [], detected_lang or "en"
        return []

    diarization_segments = _post_sortformer(audio_bytes) if diarize else []
    out_words = _merge_words_with_speakers(words, diarization_segments)

    if return_language:
        return out_words, detected_lang or "en"
    return out_words


@timeit
def deepgram_prerecorded(
    audio_url: str,
    speakers_count: int = None,
    attempts: int = 0,
    return_language: bool = False,
    diarize: bool = True,
    language: Optional[str] = None,
    model: str = "nova-3",
    keywords: Optional[Sequence[str]] = None,
) -> Union[List[dict], Tuple[List[dict], str]]:
    """
    Transcribe audio using Deepgram's pre-recorded API.
    Returns words in same format as fal_whisperx for compatibility with existing postprocessing.

    Args:
        audio_url: URL to the audio file
        speakers_count: Hint for number of speakers (not used by Deepgram, kept for API compatibility)
        attempts: Current retry attempt number
        return_language: If True, returns (words, language) tuple
        language: Language code to force, or 'multi' for multilingual auto-detection
        diarize: If True, enable speaker diarization
        keywords: Custom vocabulary words to boost transcription accuracy

    Returns:
        List of word dicts with format: {'timestamp': [start, end], 'speaker': 'SPEAKER_XX', 'text': 'word'}
        Or tuple of (words, language) if return_language=True
    """
    logger.info(f'deepgram_prerecorded {audio_url} {speakers_count} {attempts}')

    try:
        # 'multi' language means auto-detection
        is_multi = language == 'multi'
        should_detect_language = return_language or is_multi
        options = {
            "model": model,
            "smart_format": True,
            "punctuate": True,
            "diarize": diarize,
            "detect_language": should_detect_language,
            "utterances": True,
        }
        if language and not is_multi:
            options["language"] = language

        if keywords:
            if model in ('nova-3',):
                options["keyterm"] = list(keywords)
            else:
                options["keywords"] = list(keywords)

        response = (
            _deepgram_client_for_request()
            .listen.rest.v("1")
            .transcribe_url({"url": audio_url}, options, timeout=_DG_TIMEOUT)
        )

        # Extract words from response
        result = response.to_dict()
        channels = result.get('results', {}).get('channels', [])
        if not channels:
            raise Exception('No channels found in response')

        alternatives = channels[0].get('alternatives', [])
        if not alternatives:
            raise Exception('No alternatives found in response')

        dg_words = alternatives[0].get('words', [])
        if not dg_words:
            if return_language:
                detected_lang = channels[0].get('detected_language', 'en')
                if detected_lang and '-' in detected_lang:
                    detected_lang = detected_lang.split('-')[0]
                return [], detected_lang or 'en'
            return []

        # Convert Deepgram format to fal_whisperx compatible format
        # Deepgram: {word, start, end, confidence, punctuated_word, speaker (int)}
        # Expected: {timestamp: [start, end], speaker: 'SPEAKER_XX', text: 'word'}
        words = []
        for w in dg_words:
            speaker_id = w.get('speaker', 0)
            words.append(
                {
                    'timestamp': [w['start'], w['end']],
                    'speaker': f"SPEAKER_{speaker_id:02d}" if speaker_id is not None else None,
                    'text': w.get('punctuated_word', w['word']),
                }
            )

        if return_language:
            # Deepgram returns detected_language in the channel
            detected_lang = channels[0].get('detected_language', 'en')
            # Normalize language code (Deepgram might return 'en-US', we want 'en')
            if detected_lang and '-' in detected_lang:
                detected_lang = detected_lang.split('-')[0]
            return words, detected_lang or 'en'

        return words

    except Exception as e:
        logger.error(f'Deepgram prerecorded error: {e}')
        if attempts < 1:
            return deepgram_prerecorded(
                audio_url,
                speakers_count,
                attempts + 1,
                return_language,
                diarize,
                language,
                model,
                keywords,
            )
        raise RuntimeError(f'Deepgram transcription failed after {attempts + 1} attempts: {e}')


@timeit
def deepgram_prerecorded_from_bytes(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    diarize: bool = True,
    attempts: int = 0,
    encoding: Optional[str] = None,
    channels: int = 1,
    language: Optional[str] = None,
    model: str = "nova-3",
    return_language: bool = False,
    keywords: Optional[Sequence[str]] = None,
) -> Union[List[dict], Tuple[List[dict], str]]:
    """
    Transcribe audio bytes using Deepgram's pre-recorded API.
    Returns words with speaker labels when diarize=True.

    Supports both WAV format (default) and raw PCM audio.
    For raw PCM, pass encoding='linear16' with appropriate sample_rate and channels.

    Args:
        audio_bytes: Audio bytes (WAV format or raw PCM)
        sample_rate: Audio sample rate in Hz (required for raw PCM, ignored for WAV)
        diarize: If True, enable speaker diarization
        attempts: Current retry attempt number
        encoding: Audio encoding format (e.g. 'linear16' for raw PCM). None for WAV.
        channels: Number of audio channels (default 1 for mono)
        language: Language code for transcription, or None for auto-detect
        model: Deepgram model name (default 'nova-3')
        return_language: If True, returns (words, language) tuple
        keywords: Custom vocabulary words to boost transcription accuracy

    Returns:
        List of word dicts with format: {'timestamp': [start, end], 'speaker': 'SPEAKER_XX', 'text': 'word'}
        Or tuple of (words, language) if return_language=True
    """
    logger.info(
        f'deepgram_prerecorded_from_bytes bytes_len={len(audio_bytes)} {sample_rate} {diarize} {attempts} encoding={encoding} language={language} model={model}'
    )

    if STT_BATCH_BACKEND == "whisper":
        return local_whisper_prerecorded_from_bytes(
            audio_bytes,
            sample_rate=sample_rate,
            diarize=diarize,
            attempts=attempts,
            encoding=encoding,
            channels=channels,
            language=language,
            model=None,  # use WHISPER_MODEL default; callers pass nova-3 which is Deepgram-specific
            return_language=return_language,
            keywords=keywords,
        )

    try:
        is_multi = language == 'multi'
        should_detect_language = return_language or is_multi
        options = {
            "model": model,
            "smart_format": True,
            "punctuate": True,
            "diarize": diarize,
            "utterances": True,
            "detect_language": should_detect_language,
        }
        if language and not is_multi:
            options["language"] = language

        if keywords:
            if str(model).startswith("nova-3"):
                options["keyterm"] = list(keywords)
            else:
                options["keywords"] = list(keywords)

        # For raw PCM, Deepgram needs encoding + sample_rate to interpret the bytes
        if encoding:
            options["encoding"] = encoding
            options["sample_rate"] = sample_rate
            options["channels"] = channels

        # Wrap bytes in BytesIO for Deepgram client
        audio_buffer = BytesIO(audio_bytes)
        mimetype = "audio/raw" if encoding else "audio/wav"
        source = {"buffer": audio_buffer, "mimetype": mimetype}

        response = (
            _deepgram_client_for_request().listen.rest.v("1").transcribe_file(source, options, timeout=_DG_TIMEOUT)
        )

        # Extract words from response
        result = response.to_dict()
        result_channels = result.get('results', {}).get('channels', [])
        if not result_channels:
            raise Exception('No channels found in response')

        alternatives = result_channels[0].get('alternatives', [])
        if not alternatives:
            raise Exception('No alternatives found in response')

        dg_words = alternatives[0].get('words', [])
        if not dg_words:
            if return_language:
                detected_lang = result_channels[0].get('detected_language', 'en')
                if detected_lang and '-' in detected_lang:
                    detected_lang = detected_lang.split('-')[0]
                return [], detected_lang or 'en'
            return []

        # Convert Deepgram format to standard format
        # Deepgram: {word, start, end, confidence, punctuated_word, speaker (int)}
        # Expected: {timestamp: [start, end], speaker: 'SPEAKER_XX', text: 'word'}
        words = []
        for w in dg_words:
            speaker_id = w.get('speaker', 0)
            words.append(
                {
                    'timestamp': [w['start'], w['end']],
                    'speaker': f"SPEAKER_{speaker_id:02d}" if speaker_id is not None else None,
                    'text': w.get('punctuated_word', w['word']),
                }
            )

        if return_language:
            detected_lang = result_channels[0].get('detected_language', 'en')
            if detected_lang and '-' in detected_lang:
                detected_lang = detected_lang.split('-')[0]
            return words, detected_lang or 'en'

        return words

    except Exception as e:
        logger.error(f'Deepgram prerecorded from bytes error: {e}')
        if attempts < 1:
            return deepgram_prerecorded_from_bytes(
                audio_bytes,
                sample_rate,
                diarize,
                attempts + 1,
                encoding,
                channels,
                language,
                model,
                return_language,
                keywords,
            )
        raise RuntimeError(f'Deepgram transcription failed after {attempts + 1} attempts: {e}')


@timeit
def fal_whisperx(
    audio_url: str,
    speakers_count: int = None,
    attempts: int = 0,
    return_language: bool = False,
    diarize: bool = True,
    chunk_level: str = 'word',
) -> List[dict]:
    logger.info(f'fal_whisperx {audio_url} {speakers_count} {attempts}')

    try:
        handler = fal_client.submit(
            "fal-ai/whisper",
            arguments={
                "audio_url": audio_url,
                'task': 'transcribe',
                'diarize': diarize,
                'chunk_level': chunk_level,
                'version': '3',
                'batch_size': 64,
                'num_speakers': speakers_count,
            },
        )
        result = handler.get()
        # print(result)
        words = result.get('chunks', [])
        if not words:
            raise Exception('No chunks found')
        if return_language:
            languages = result.get('inferred_languages', ['en'])
            language = languages[0] if languages else 'en'
            return words, language
        return words
    except Exception as e:
        logger.error(e)
        if attempts < 2:
            return fal_whisperx(audio_url, speakers_count, attempts + 1, return_language)
        if return_language:
            return [], 'en'
        return []


def _words_cleaning(words: List[dict]):
    words_cleaned: List[dict] = []
    for i, w in enumerate(words):
        # if w['timestamp'][0] == w['timestamp'][1]:
        #     continue
        words_cleaned.append(
            {
                'start': round(w['timestamp'][0], 2),
                'end': round(w['timestamp'][1] or w['timestamp'][0] + 1, 2),
                'speaker': w['speaker'],
                'text': str(w['text']).strip(),
                'is_user': False,
                'person_id': None,
            }
        )

    for i, word in enumerate(words_cleaned):
        speaker = word['speaker']
        if not speaker:
            prev_chunk = words_cleaned[i - 1] if i > 0 else None
            next_chunk = words_cleaned[i + 1] if i < len(words_cleaned) - 1 else None
            prev_speaker = prev_chunk['speaker'] if prev_chunk else None
            next_speaker = next_chunk['speaker'] if next_chunk else None

            if prev_speaker and next_speaker:
                if prev_speaker == next_speaker:
                    speaker = prev_chunk['speaker']
                else:
                    secs_from_prev = word['start'] - prev_chunk['end'] if prev_chunk else 0
                    secs_to_next = next_chunk['start'] - word['end'] if next_chunk else 0
                    speaker = prev_speaker if secs_from_prev < secs_to_next else next_speaker
            elif prev_speaker:
                speaker = prev_speaker
            elif next_speaker:
                speaker = next_speaker
            else:
                speaker = 'SPEAKER_00'

            words_cleaned[i]['speaker'] = speaker

    # for chunk in words_cleaned:
    #     print(chunk)
    return words_cleaned


def _retrieve_user_speaker_id(words: list, skip_n_seconds: int):
    if not skip_n_seconds:
        return None

    user_speaker_id = defaultdict(int)
    for word in words:
        if word['start'] >= skip_n_seconds:
            break
        if not word['speaker']:
            continue
        user_speaker_id[word['speaker']] += 1

    user_speaker_id = max(user_speaker_id, key=user_speaker_id.get) if user_speaker_id else None
    return user_speaker_id


def _merge_segments(words: List[dict], skip_n_seconds: int, user_speaker_id: str):
    segments = []
    for word in words:
        if word['start'] < skip_n_seconds:
            continue
        word['is_user'] = word['speaker'] == user_speaker_id if word['speaker'] else False

        same_prev_speaker = word['speaker'] == segments[-1]['speaker'] if segments else False
        seconds_from_prev = word['start'] - segments[-1]['end'] if segments else 0

        # TODO: consider having a max segment size too
        if segments and same_prev_speaker and seconds_from_prev < 30:
            segments[-1]['end'] = word['end']
            segments[-1]['text'] += ' ' + word['text']
        else:
            segments.append(word)
    return segments


def _segments_as_objects(segments: List[dict]) -> List[TranscriptSegment]:
    if not segments:
        return []
    starts_at = segments[0]['start']
    return [
        TranscriptSegment(
            text=str(segment['text']).strip().capitalize(),
            speaker=segment['speaker'],
            is_user=segment['is_user'],
            person_id=None,
            start=round(segment['start'] - starts_at, 2),
            end=round(segment['end'] - starts_at, 2),
        )
        for segment in segments
    ]


def postprocess_words(
    words: List[dict], duration: int, skip_n_seconds: int = 0  # , merge_segments: bool = True
) -> List[TranscriptSegment]:
    words: List[dict] = _words_cleaning(words)
    user_speaker_id = _retrieve_user_speaker_id(words, skip_n_seconds)
    segments = _merge_segments(words, skip_n_seconds, user_speaker_id)
    segments = _segments_as_objects(segments)
    return segments
