"""
TTS (Text-to-Speech) service for text to audio synthesis.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile

from app.config import settings
from app.models.tts import (
    TTSCacheEntry,
    TTSModelInfo,
    TTSVoiceInfo,
)


DEFAULT_TTS_MODEL = "Supertone/supertonic-3"
DEFAULT_TTS_VOICE = "M1"
DEFAULT_TTS_LANGUAGE = "en"
SUPERTONIC_SAMPLE_RATE = 44100
SUPERTONIC_TOTAL_STEPS = 8

SUPERTONIC_LANGUAGES = [
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
    "na",
]

LANGUAGE_ALIASES = {
    "auto": "na",
    "unknown": "na",
    "fallback": "na",
    "eng": "en",
    "english": "en",
    "kor": "ko",
    "korean": "ko",
    "jpn": "ja",
    "japanese": "ja",
    "ara": "ar",
    "arabic": "ar",
    "bulgarian": "bg",
    "czech": "cs",
    "danish": "da",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "greek": "el",
    "spa": "es",
    "spanish": "es",
    "estonian": "et",
    "finnish": "fi",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "hin": "hi",
    "hindi": "hi",
    "croatian": "hr",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "lithuanian": "lt",
    "latvian": "lv",
    "dutch": "nl",
    "polish": "pl",
    "por": "pt",
    "portuguese": "pt",
    "romanian": "ro",
    "rus": "ru",
    "russian": "ru",
    "slovak": "sk",
    "slovenian": "sl",
    "swedish": "sv",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
}

SUPERTONIC_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

TTS_MODELS = {
    DEFAULT_TTS_MODEL: {
        "name": "Supertonic 3",
        "description": "Lightning-fast on-device multilingual TTS using ONNX Runtime.",
        "sample_rate": SUPERTONIC_SAMPLE_RATE,
        "languages": SUPERTONIC_LANGUAGES,
        "model_family": "supertonic",
        "supports_instructions": False,
        "supports_voice_presets": True,
        "requires_reference_audio": False,
        "features": [
            "31 language codes plus unknown-language fallback",
            "Built-in voice styles M1-M5 and F1-F5",
            "ONNX Runtime local inference with no cloud call",
            "Expression tags such as <laugh>, <breath>, and <sigh>",
        ],
    },
}

VOICE_OPTIONS = {
    voice_id: {
        "name": f"Supertonic {voice_id}",
        "language": "multilingual",
        "model_family": "supertonic",
        "description": "Built-in Supertonic 3 voice style",
        "native_language": "Multilingual",
    }
    for voice_id in SUPERTONIC_VOICES
}

tts_cache = {}
TTS_INDEX_PATH = settings.upload_dir / "tts_index.json"
TTS_MODEL_CACHE = {}


class TTSBackendUnavailableError(RuntimeError):
    """Raised when a configured TTS backend is not installed or configured."""


def persist_tts_cache() -> None:
    """Persist TTS metadata so Archive survives backend restarts."""
    TTS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = TTS_INDEX_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps([entry.model_dump() for entry in tts_cache.values()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(TTS_INDEX_PATH)


def load_tts_cache() -> None:
    """Load persisted TTS metadata, falling back to saved WAV/MP3 files."""
    tts_cache.clear()

    if TTS_INDEX_PATH.exists():
        try:
            entries = json.loads(TTS_INDEX_PATH.read_text(encoding="utf-8"))
            for entry in entries:
                if isinstance(entry, dict) and entry.get("id"):
                    tts_cache[entry["id"]] = TTSCacheEntry(**entry)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            tts_cache.clear()

    recovered_any = False
    for audio_path in settings.tts_output_dir.iterdir():
        if not audio_path.is_file() or audio_path.suffix.lower() not in {".wav", ".mp3"}:
            continue
        tts_id = audio_path.stem
        if tts_id in tts_cache:
            continue
        tts_cache[tts_id] = TTSCacheEntry(
            id=tts_id,
            text=f"Recovered TTS audio: {audio_path.name}",
            model="Recovered from uploads",
            voice="unknown",
            speed=1.0,
            pitch=1.0,
            language=None,
            instruction=None,
            duration=0,
            sample_rate=0,
            created_at=datetime.fromtimestamp(audio_path.stat().st_mtime).isoformat(),
        )
        recovered_any = True

    if recovered_any or not TTS_INDEX_PATH.exists():
        persist_tts_cache()


def normalize_language_code(language: Optional[str]) -> str:
    """Normalize user-facing language input into a Supertonic language code."""
    normalized = (language or "").strip().lower()
    if not normalized:
        return DEFAULT_TTS_LANGUAGE
    normalized = LANGUAGE_ALIASES.get(normalized, normalized)
    return normalized if normalized in SUPERTONIC_LANGUAGES else "na"


def detect_language(text: str) -> str:
    """Best-effort language detection for choosing a Supertonic language code."""
    if any("\uac00" <= c <= "\ud7af" for c in text):
        return "ko"
    if any("\u3040" <= c <= "\u30ff" for c in text):
        return "ja"
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return "na"
    return DEFAULT_TTS_LANGUAGE


def get_model_family(model_name: str) -> str:
    """Return the voice preset family for a TTS model."""
    return TTS_MODELS.get(model_name, {}).get("model_family", "supertonic")


def default_voice_for_model(_model_name: str) -> str:
    """Return the default speaker for the selected TTS model."""
    return DEFAULT_TTS_VOICE


def is_voice_compatible(model_name: str, voice: str) -> bool:
    """Return whether a voice preset can be used with the selected model."""
    return model_name == DEFAULT_TTS_MODEL and voice in VOICE_OPTIONS


def load_tts_model(model_name: str):
    """Load the Supertonic 3 TTS engine."""
    from fastapi import HTTPException

    if model_name in TTS_MODEL_CACHE:
        return TTS_MODEL_CACHE[model_name]

    model_config = TTS_MODELS.get(model_name)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Unknown TTS model: {model_name}")

    try:
        try:
            from supertonic import TTS
        except ImportError as exc:
            raise TTSBackendUnavailableError(
                "Supertonic backend is not installed. Install it with "
                "`pip install supertonic` or `pip install -r requirements.txt`, "
                "restart the API, then try again."
            ) from exc

        model = TTS(auto_download=True)
        TTS_MODEL_CACHE[model_name] = {
            "model": model,
            "config": model_config,
            "family": "supertonic",
            "loaded": True,
        }
        return TTS_MODEL_CACHE[model_name]

    except TTSBackendUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load TTS model: {str(exc)}")


def _normalize_generated_audio(audio) -> np.ndarray:
    """Convert model output to a 1-D float32 numpy array in -1..1 range."""
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.ndim > 1:
        audio = audio.reshape(-1)

    if np.issubdtype(audio.dtype, np.integer):
        max_value = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _duration_to_float(duration, audio_array: np.ndarray, sample_rate: int) -> float:
    """Normalize Supertonic duration output, falling back to audio length."""
    try:
        duration_value = float(np.asarray(duration).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        duration_value = 0.0

    if duration_value <= 0 and sample_rate > 0:
        duration_value = float(audio_array.shape[0] / sample_rate)
    return duration_value


def _synthesize_with_supertonic(
    model,
    text: str,
    voice: str,
    language: str,
    speed: float,
) -> tuple[np.ndarray, int, float]:
    """Run Supertonic 3 inference."""
    style = model.get_voice_style(voice_name=voice)
    audio, duration = model.synthesize(
        text=text,
        voice_style=style,
        lang=language,
        speed=speed,
        total_steps=SUPERTONIC_TOTAL_STEPS,
    )
    audio_array = _normalize_generated_audio(audio)
    duration_seconds = _duration_to_float(duration, audio_array, SUPERTONIC_SAMPLE_RATE)
    return audio_array, SUPERTONIC_SAMPLE_RATE, duration_seconds


def synthesize_audio(
    text: str,
    model_name: str,
    voice: str = DEFAULT_TTS_VOICE,
    speed: float = 1.0,
    pitch: float = 1.0,
    language: Optional[str] = None,
    instruction: Optional[str] = None,
) -> tuple[np.ndarray, int, float]:
    """
    Synthesize speech from text using Supertonic 3.

    The pitch and instruction arguments are accepted for API compatibility but
    are not treated as prompts by Supertonic.
    """
    from fastapi import HTTPException

    model_data = load_tts_model(model_name or DEFAULT_TTS_MODEL)
    normalized_language = normalize_language_code(language) if language else detect_language(text)
    normalized_voice = (voice or "").strip() or DEFAULT_TTS_VOICE
    if not is_voice_compatible(model_name or DEFAULT_TTS_MODEL, normalized_voice):
        normalized_voice = DEFAULT_TTS_VOICE

    try:
        audio_array, sample_rate, duration = _synthesize_with_supertonic(
            model=model_data["model"],
            text=text,
            voice=normalized_voice,
            language=normalized_language,
            speed=speed,
        )

        if audio_array.size == 0:
            raise HTTPException(status_code=502, detail="TTS model returned an empty audio array.")

        peak = float(np.max(np.abs(audio_array)))
        if peak < 0.001:
            raise HTTPException(status_code=502, detail="TTS model returned silent audio.")
        if peak > 1.0:
            audio_array = audio_array / peak

        if duration <= 0:
            duration = float(audio_array.shape[0] / sample_rate)
        if duration <= 0:
            raise HTTPException(status_code=502, detail="TTS model returned zero-duration audio.")

        return audio_array, sample_rate, duration

    except TTSBackendUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(exc)}")


def save_audio_to_file(
    audio_data: np.ndarray,
    sample_rate: int,
    tts_id: str,
    output_format: str = "wav",
) -> str:
    """Save audio data to a WAV or MP3 file."""
    output_dir = settings.tts_output_dir
    file_path = output_dir / f"{tts_id}.{output_format}"
    safe_audio = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    if safe_audio.size == 0 or float(np.max(np.abs(safe_audio))) < 0.001:
        raise ValueError("Cannot save empty or silent TTS audio")

    pcm_audio = (np.clip(safe_audio, -1.0, 1.0) * 32767).astype(np.int16)
    wav_path = output_dir / f"{tts_id}.wav"
    wavfile.write(str(wav_path), sample_rate, pcm_audio)

    if output_format == "wav":
        return str(wav_path)

    from pydub import AudioSegment

    AudioSegment.from_file(wav_path, format="wav").export(file_path, format="mp3")
    if wav_path != file_path:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    return str(file_path)


def get_available_models() -> list[TTSModelInfo]:
    """Get list of available TTS models."""
    return [
        TTSModelInfo(
            id=model_id,
            name=config["name"],
            description=config["description"],
            sample_rate=config["sample_rate"],
            languages=config["languages"],
            model_family=config["model_family"],
            supports_instructions=config["supports_instructions"],
            supports_voice_presets=config["supports_voice_presets"],
            requires_reference_audio=config["requires_reference_audio"],
            features=config["features"],
        )
        for model_id, config in TTS_MODELS.items()
    ]


def get_available_voices() -> list[TTSVoiceInfo]:
    """Get list of available voices."""
    return [
        TTSVoiceInfo(
            id=voice_id,
            name=config["name"],
            language=config["language"],
            model_family=config["model_family"],
            description=config["description"],
            native_language=config["native_language"],
        )
        for voice_id, config in VOICE_OPTIONS.items()
    ]


async def process_tts(
    text: str,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_TTS_VOICE,
    speed: float = 1.0,
    pitch: float = 1.0,
    language: Optional[str] = None,
    instruction: Optional[str] = None,
    output_format: str = "wav",
) -> tuple[str, bytes, float, int]:
    """Process a TTS request."""
    from fastapi import HTTPException

    normalized_text = (text or "").strip()
    if not normalized_text:
        raise HTTPException(status_code=400, detail="Text is required for TTS synthesis.")

    if len(normalized_text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long. Maximum 5000 characters.")

    if speed < 0.7 or speed > 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.7 and 2.0")

    if pitch < 0.5 or pitch > 2.0:
        raise HTTPException(status_code=400, detail="Pitch must be between 0.5 and 2.0")

    if model not in TTS_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    normalized_language = normalize_language_code(language)
    normalized_voice = (voice or "").strip() or DEFAULT_TTS_VOICE
    if not is_voice_compatible(model, normalized_voice):
        normalized_voice = DEFAULT_TTS_VOICE

    tts_id = str(uuid.uuid4())

    audio_data, sample_rate, duration = synthesize_audio(
        text=normalized_text,
        model_name=model,
        voice=normalized_voice,
        speed=speed,
        pitch=pitch,
        language=normalized_language,
        instruction=instruction,
    )

    file_path = save_audio_to_file(
        audio_data=audio_data,
        sample_rate=sample_rate,
        tts_id=tts_id,
        output_format=output_format,
    )

    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    tts_cache[tts_id] = TTSCacheEntry(
        id=tts_id,
        text=normalized_text,
        model=model,
        voice=normalized_voice,
        speed=speed,
        pitch=pitch,
        language=normalized_language,
        instruction=None,
        duration=duration,
        sample_rate=sample_rate,
        created_at=datetime.now().isoformat(),
    )
    persist_tts_cache()

    return tts_id, audio_bytes, duration, sample_rate


load_tts_cache()
