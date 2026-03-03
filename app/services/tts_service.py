"""
TTS (Text-to-Speech) service for text to audio synthesis
"""

import os
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
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


# TTS Models configuration
QWEN3_TTS_MODELS = {
    "qwen3-tts-0.6b": {
        "name": "Qwen3-TTS-0.6B",
        "description": "Lightweight TTS model, fast inference",
        "sample_rate": 24000,
        "languages": ["zh", "en"],
    },
    "qwen3-tts-1.8b": {
        "name": "Qwen3-TTS-1.8B",
        "description": "Standard TTS model, balanced quality and speed",
        "sample_rate": 24000,
        "languages": ["zh", "en"],
    },
    "qwen3-tts-4b": {
        "name": "Qwen3-TTS-4B",
        "description": "High-quality TTS model, best voice quality",
        "sample_rate": 24000,
        "languages": ["zh", "en"],
    },
    "cosyvoice-300m": {
        "name": "CosyVoice-300M",
        "description": "CosyVoice model for natural speech synthesis",
        "sample_rate": 22050,
        "languages": ["zh", "en"],
    },
    "cosyvoice-300m-sft": {
        "name": "CosyVoice-300M-SFT",
        "description": "CosyVoice SFT model with fine-tuned voices",
        "sample_rate": 22050,
        "languages": ["zh", "en"],
    },
    "cosyvoice-300m-instruct": {
        "name": "CosyVoice-300M-Instruct",
        "description": "CosyVoice Instruct model for controllable synthesis",
        "sample_rate": 22050,
        "languages": ["zh", "en"],
    },
}

# Available voice options
VOICE_OPTIONS = {
    "default": {"name": "Default", "language": "auto"},
    "male-1": {"name": "Male Voice 1", "language": "zh"},
    "male-2": {"name": "Male Voice 2", "language": "zh"},
    "female-1": {"name": "Female Voice 1", "language": "zh"},
    "female-2": {"name": "Female Voice 2", "language": "zh"},
    "english-male": {"name": "English Male", "language": "en"},
    "english-female": {"name": "English Female", "language": "en"},
}

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In-memory cache for TTS results
tts_cache = {}

# Model cache
TTS_MODEL_CACHE = {}


def load_tts_model(model_name: str):
    """
    Load TTS model by name.
    Supports Qwen3-TTS and CosyVoice models.

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

    model_config = QWEN3_TTS_MODELS.get(model_name)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Unknown TTS model: {model_name}")

    try:
        if "qwen3" in model_name:
            # Qwen3-TTS model loading
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = f"Qwen/{model_name}"

            # In production, uncomment below:
            # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_path,
            #     torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            #     device_map="auto",
            #     trust_remote_code=True
            # )
            # TTS_MODEL_CACHE[model_name] = {
            #     "model": model,
            #     "tokenizer": tokenizer,
            #     "config": model_config,
            #     "loaded": True
            # }

            # Placeholder for demo
            TTS_MODEL_CACHE[model_name] = {
                "model": None,
                "tokenizer": None,
                "config": model_config,
                "loaded": True
            }

        elif "cosyvoice" in model_name:
            # CosyVoice model loading
            # from cosyvoice import CosyVoice
            # cosyvoice = CosyVoice(model_name)
            # TTS_MODEL_CACHE[model_name] = {
            #     "model": cosyvoice,
            #     "config": model_config,
            #     "loaded": True
            # }

            # Placeholder for demo
            TTS_MODEL_CACHE[model_name] = {
                "model": None,
                "config": model_config,
                "loaded": True
            }

        return TTS_MODEL_CACHE[model_name]

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
        return "zh"
    return "en"


def synthesize_audio(
    text: str,
    model_name: str,
    voice: str = "default",
    speed: float = 1.0,
    pitch: float = 1.0,
    language: Optional[str] = None,
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
        if model_data.get("loaded"):
            # Placeholder implementation
            # In production, use actual model inference:
            # if "qwen3" in model_name:
            #     audio = model_data["model"].tts(text, speaker=voice, speed=speed)
            # elif "cosyvoice" in model_name:
            #     audio = model_data["model"].inference_sft(text, voice)

            sample_rate = model_config["sample_rate"]
            duration = len(text) * 0.15 / speed  # Rough estimate
            num_samples = int(sample_rate * duration)

            # Generate placeholder audio (silence for demo)
            audio_array = np.zeros(num_samples, dtype=np.float32)

            return audio_array, sample_rate, duration

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

    if output_format == "wav":
        wavfile.write(str(file_path), sample_rate, (audio_data * 32767).astype(np.int16))
    else:
        # For MP3, we'd need additional processing
        # For now, save as WAV and rename
        wav_path = output_dir / f"{tts_id}.wav"
        wavfile.write(str(wav_path), sample_rate, (audio_data * 32767).astype(np.int16))
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
            languages=config["languages"]
        )
        for model_id, config in QWEN3_TTS_MODELS.items()
    ]


def get_available_voices() -> list[TTSVoiceInfo]:
    """Get list of available voices"""
    return [
        TTSVoiceInfo(
            id=voice_id,
            name=config["name"],
            language=config["language"]
        )
        for voice_id, config in VOICE_OPTIONS.items()
    ]


async def process_tts(
    text: str,
    model: str = "qwen3-tts-1.8b",
    voice: str = "default",
    speed: float = 1.0,
    pitch: float = 1.0,
    language: Optional[str] = None,
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
        output_format: Output format

    Returns:
        Tuple of (tts_id, audio_bytes, duration, sample_rate)

    Raises:
        HTTPException: If processing fails
    """
    from fastapi import HTTPException

    # Validate inputs
    if len(text) > 5000:
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

    if model not in QWEN3_TTS_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    if voice not in VOICE_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")

    # Generate unique ID
    tts_id = str(uuid.uuid4())

    # Synthesize speech
    audio_data, sample_rate, duration = synthesize_audio(
        text=text,
        model_name=model,
        voice=voice,
        speed=speed,
        pitch=pitch,
        language=language,
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
        text=text,
        model=model,
        voice=voice,
        speed=speed,
        pitch=pitch,
        language=language,
        duration=duration,
        sample_rate=sample_rate,
        created_at=datetime.now().isoformat()
    )

    return tts_id, audio_bytes, duration, sample_rate
