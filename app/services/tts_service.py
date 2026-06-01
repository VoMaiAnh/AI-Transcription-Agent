"""
TTS (Text-to-Speech) service for text to audio synthesis
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
import torch

from app.config import settings
from app.models.tts import (
    TTSModelInfo,
    TTSVoiceInfo,
    TTSCacheEntry,
)


DEFAULT_TTS_MODEL = "k2-fsa/OmniVoice"
DEFAULT_TTS_VOICE = "voice-design"

OMNIVOICE_LANGUAGES = [
    "Auto",
    "English",
    "Chinese",
    "Spanish",
    "Arabic",
    "Hindi",
    "French",
    "German",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "600+ languages",
]

# TTS Models configuration
TTS_MODELS = {
    "k2-fsa/OmniVoice": {
        "name": "OmniVoice",
        "description": "Massively multilingual zero-shot TTS model with voice cloning and no-reference voice design.",
        "sample_rate": 24000,
        "languages": OMNIVOICE_LANGUAGES,
        "model_family": "omnivoice",
        "supports_instructions": True,
        "supports_voice_presets": False,
        "requires_reference_audio": False,
        "features": [
            "600+ language zero-shot TTS coverage",
            "Voice design through speaker attributes with no reference audio required",
            "Voice cloning support when a reference-audio route is added",
            "Fine-grained non-verbal symbols and pronunciation correction",
            "Fast diffusion language-model style inference with reported RTF as low as 0.025",
        ],
    },
}

VOICE_OPTIONS = {}

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In-memory cache for TTS results
tts_cache = {}
TTS_INDEX_PATH = settings.upload_dir / "tts_index.json"

# Model cache
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


def load_tts_model(model_name: str):
    """
    Load TTS model by name.
    Supports OmniVoice models.

    Args:
        model_name: Name of the TTS model to load

    Returns:
        Loaded model data

    Raises:
        HTTPException: If model loading fails
    """
    from fastapi import HTTPException

    if model_name in TTS_MODEL_CACHE:
        return TTS_MODEL_CACHE[model_name]

    model_config = TTS_MODELS.get(model_name)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Unknown TTS model: {model_name}")

    try:
        if model_config["model_family"] == "omnivoice":
            try:
                from omnivoice import OmniVoice
            except ImportError as exc:
                raise TTSBackendUnavailableError(
                    "OmniVoice backend is not installed. Install it with "
                    "`pip install -r requirements.txt`, restart the API, then try again."
                ) from exc

            load_kwargs = {
                "device_map": "cuda:0" if DEVICE == "cuda" else "cpu",
                "dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
            }
            model = OmniVoice.from_pretrained(model_name, **load_kwargs)
            TTS_MODEL_CACHE[model_name] = {
                "model": model,
                "config": model_config,
                "family": "omnivoice",
                "loaded": True,
            }

        return TTS_MODEL_CACHE[model_name]

    except TTSBackendUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load TTS model: {str(e)}"
        )


def detect_language(text: str) -> str:
    """
    Auto-detect language from text

    Args:
        text: Input text

    Returns:
        Language code ('zh' or 'en')
    """
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return "Chinese"
    return "English"


def get_model_family(model_name: str) -> str:
    """Return the voice preset family for a TTS model."""
    config = TTS_MODELS.get(model_name, {})
    return config.get("model_family", "omnivoice")


def is_voice_compatible(model_name: str, voice: str) -> bool:
    """Return whether a voice preset can be used with the selected model."""
    if get_model_family(model_name) == "omnivoice":
        # OmniVoice voice design uses the instruction field instead of preset voices.
        return True

    return False


def _normalize_generated_audio(audio) -> np.ndarray:
    """Convert model output to a 1-D float32 numpy array."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    else:
        audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.ndim > 1:
        audio = audio.reshape(-1)

    return np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _synthesize_with_omnivoice(
    model,
    text: str,
    instruction: Optional[str],
) -> tuple[np.ndarray, int]:
    """Run real OmniVoice voice-design inference."""
    generate_kwargs = {"text": text}
    if instruction:
        generate_kwargs["instruct"] = instruction

    audio = model.generate(**generate_kwargs)
    if not audio:
        raise TTSBackendUnavailableError("OmniVoice returned no audio.")

    return _normalize_generated_audio(audio[0]), 24000


def _compose_omnivoice_instruction(
    instruction: Optional[str],
    pitch: float,
) -> Optional[str]:
    """Fold the pitch control into OmniVoice voice-design attributes."""
    parts = []
    normalized = (instruction or "").strip()
    if normalized:
        parts.append(normalized)

    lower_instruction = normalized.lower()
    if "pitch" not in lower_instruction:
        if pitch <= 0.7:
            parts.append("very low pitch")
        elif pitch < 0.95:
            parts.append("low pitch")
        elif pitch > 1.3:
            parts.append("very high pitch")
        elif pitch > 1.05:
            parts.append("high pitch")

    return ", ".join(parts) or None


def synthesize_audio(
    text: str,
    model_name: str,
    voice: str = DEFAULT_TTS_VOICE,
    speed: float = 1.0,
    pitch: float = 1.0,
    language: Optional[str] = None,
    instruction: Optional[str] = None,
) -> tuple:
    """
    Synthesize speech from text using the specified TTS model.

    Args:
        text: Text to synthesize
        model_name: TTS model to use
        voice: Voice to use
        speed: Speech speed (0.5-2.0)
        pitch: Pitch adjustment (0.5-2.0)
        language: Language code (auto-detected if None)
        instruction: Optional natural-language control prompt

    Returns:
        Tuple of (audio_array, sample_rate, duration_seconds)

    Raises:
        HTTPException: If synthesis fails
    """
    from fastapi import HTTPException

    model_data = load_tts_model(model_name)
    model_config = model_data["config"]

    # Determine language
    if not language:
        language = detect_language(text)

    try:
        if not model_data.get("loaded") or model_data.get("model") is None:
            raise TTSBackendUnavailableError(
                f"TTS model '{model_name}' is not loaded. Real synthesis is unavailable."
            )

        if model_config["model_family"] == "omnivoice":
            audio_array, sample_rate = _synthesize_with_omnivoice(
                model=model_data["model"],
                text=text,
                instruction=_compose_omnivoice_instruction(instruction, pitch),
            )
        else:
            raise TTSBackendUnavailableError(
                f"Real synthesis for model family '{model_config['model_family']}' is not implemented."
            )

        if audio_array.size == 0:
            raise HTTPException(status_code=502, detail="TTS model returned an empty audio array.")

        peak = float(np.max(np.abs(audio_array)))
        if peak < 0.001:
            raise HTTPException(status_code=502, detail="TTS model returned silent audio.")

        if peak > 1.0:
            audio_array = audio_array / peak

        duration = float(audio_array.shape[0] / sample_rate)
        if duration <= 0:
            raise HTTPException(status_code=502, detail="TTS model returned zero-duration audio.")

        return audio_array, sample_rate, duration

    except TTSBackendUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {str(e)}"
        )


def save_audio_to_file(
    audio_data: np.ndarray,
    sample_rate: int,
    tts_id: str,
    output_format: str = "wav"
) -> str:
    """
    Save audio data to file

    Args:
        audio_data: Audio data array
        sample_rate: Sample rate in Hz
        tts_id: Unique ID for the file
        output_format: Output format (wav or mp3)

    Returns:
        Path to saved file
    """
    output_dir = settings.tts_output_dir
    file_path = output_dir / f"{tts_id}.{output_format}"
    safe_audio = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    if safe_audio.size == 0 or float(np.max(np.abs(safe_audio))) < 0.001:
        raise ValueError("Cannot save empty or silent TTS audio")
    pcm_audio = (np.clip(safe_audio, -1.0, 1.0) * 32767).astype(np.int16)

    if output_format == "wav":
        wavfile.write(str(file_path), sample_rate, pcm_audio)
    else:
        # For MP3, we'd need additional processing
        # For now, save as WAV and rename
        wav_path = output_dir / f"{tts_id}.wav"
        wavfile.write(str(wav_path), sample_rate, pcm_audio)
        if wav_path != file_path:
            os.rename(wav_path, file_path)

    return str(file_path)


def get_available_models() -> list[TTSModelInfo]:
    """Get list of available TTS models"""
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
    """Get list of available voices"""
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
    output_format: str = "wav"
) -> tuple:
    """
    Process TTS request

    Args:
        text: Text to synthesize
        model: TTS model to use
        voice: Voice to use
        speed: Speech speed
        pitch: Pitch adjustment
        language: Language code
        instruction: Optional natural-language control prompt
        output_format: Output format

    Returns:
        Tuple of (tts_id, audio_bytes, duration, sample_rate)

    Raises:
        HTTPException: If processing fails
    """
    from fastapi import HTTPException

    # Validate inputs
    normalized_text = (text or "").strip()
    if not normalized_text:
        raise HTTPException(status_code=400, detail="Text is required for TTS synthesis.")

    if len(normalized_text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Maximum 5000 characters."
        )

    if speed < 0.5 or speed > 2.0:
        raise HTTPException(
            status_code=400,
            detail="Speed must be between 0.5 and 2.0"
        )

    if pitch < 0.5 or pitch > 2.0:
        raise HTTPException(
            status_code=400,
            detail="Pitch must be between 0.5 and 2.0"
        )

    if model not in TTS_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    model_family = get_model_family(model)
    normalized_voice = (voice or "").strip()

    if model_family == "omnivoice":
        normalized_voice = DEFAULT_TTS_VOICE

    if not is_voice_compatible(model, normalized_voice):
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{normalized_voice}' is not available for model '{model}'"
        )

    # Generate unique ID
    tts_id = str(uuid.uuid4())

    # Synthesize speech
    normalized_language = (language or "").strip() or None
    if normalized_language and normalized_language.lower() == "auto":
        normalized_language = None

    audio_data, sample_rate, duration = synthesize_audio(
        text=normalized_text,
        model_name=model,
        voice=normalized_voice or "zero-shot",
        speed=speed,
        pitch=pitch,
        language=normalized_language,
        instruction=instruction,
    )

    # Save to file
    file_path = save_audio_to_file(
        audio_data=audio_data,
        sample_rate=sample_rate,
        tts_id=tts_id,
        output_format=output_format
    )

    # Read file bytes
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    # Cache the result
    tts_cache[tts_id] = TTSCacheEntry(
        id=tts_id,
        text=normalized_text,
        model=model,
        voice=normalized_voice or "zero-shot",
        speed=speed,
        pitch=pitch,
        language=normalized_language,
        instruction=instruction,
        duration=duration,
        sample_rate=sample_rate,
        created_at=datetime.now().isoformat()
    )
    persist_tts_cache()

    return tts_id, audio_bytes, duration, sample_rate


load_tts_cache()
