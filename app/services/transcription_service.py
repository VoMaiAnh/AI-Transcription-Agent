"""
Transcription service for audio/video to text conversion
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import whisper
from pydub import AudioSegment

from app.config import settings
from app.models.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
    STTModelInfo,
)


# STT Models configuration
STT_MODELS = {
    # Whisper models
    "whisper-tiny": {
        "type": "whisper",
        "name": "Whisper Tiny",
        "description": "Fastest, lowest accuracy"
    },
    "whisper-base": {
        "type": "whisper",
        "name": "Whisper Base",
        "description": "Fast, decent accuracy"
    },
    "whisper-small": {
        "type": "whisper",
        "name": "Whisper Small",
        "description": "Balanced speed and accuracy"
    },
    "whisper-medium": {
        "type": "whisper",
        "name": "Whisper Medium",
        "description": "Good accuracy, slower"
    },
    "whisper-large": {
        "type": "whisper",
        "name": "Whisper Large",
        "description": "Best accuracy, slowest"
    },
    # Qwen3-ASR models
    "qwen3-asr-0.6b": {
        "type": "qwen3-asr",
        "name": "Qwen3-ASR-0.6B",
        "description": "Lightweight ASR, 30+ languages, streaming support"
    },
    "qwen3-asr-1.7b": {
        "type": "qwen3-asr",
        "name": "Qwen3-ASR-1.7B",
        "description": "Standard ASR, 30+ languages + 22 Chinese dialects"
    },
}

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model caches
MODEL_CACHE = {}
QWEN3_ASR_CACHE = {}

# In-memory storage for transcription results
transcription_cache = {}


def safe_remove_file(file_path: str, max_retries: int = 3, delay: float = 0.5) -> bool:
    """
    Safely remove a file with retries (for Windows file locking issues)

    Args:
        file_path: Path to file to remove
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        True if file was removed, False otherwise
    """
    import time

    path = Path(file_path)
    if not path.exists():
        return True

    for attempt in range(max_retries):
        try:
            os.remove(path)
            return True
        except (PermissionError, OSError):
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"Warning: Could not remove file {path} after {max_retries} attempts")
                return False

    return False


def load_whisper_model(model_size: Optional[str] = None):
    """
    Load the Whisper model with caching

    Args:
        model_size: Size of the Whisper model to load

    Returns:
        Loaded Whisper model
    """
    if model_size and model_size.startswith("whisper-"):
        model_size = model_size.replace("whisper-", "")
    model_size = model_size or settings.WHISPER_MODEL
    cache_key = f"whisper-{model_size}"

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    model = whisper.load_model(model_size, device=DEVICE)
    MODEL_CACHE[cache_key] = model

    return model


def load_qwen3_asr_model(model_name: Optional[str] = None):
    """
    Load Qwen3-ASR model with caching.
    Requires: pip install qwen-asr

    Args:
        model_name: Name of the Qwen3-ASR model to load

    Returns:
        Loaded Qwen3-ASR model

    Raises:
        HTTPException: If model loading fails
    """
    from fastapi import HTTPException

    model_name = model_name or settings.QWEN3_ASR_MODEL

    # Normalize model name
    if model_name in ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]:
        model_name = f"Qwen/Qwen3-ASR-{model_name.split('-')[-1].upper()}"

    cache_key = model_name

    if cache_key in QWEN3_ASR_CACHE:
        return QWEN3_ASR_CACHE[cache_key]

    try:
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="cuda:0" if DEVICE == "cuda" else "cpu",
            max_inference_batch_size=32,
            max_new_tokens=256,
        )

        QWEN3_ASR_CACHE[cache_key] = model
        return model

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Qwen3-ASR not installed. Install with: pip install qwen-asr"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load Qwen3-ASR model: {str(e)}"
        )


def convert_to_wav(file_path: str) -> str:
    """
    Convert audio file to WAV format (16kHz mono)

    Args:
        file_path: Path to audio file

    Returns:
        Path to converted WAV file

    Raises:
        HTTPException: If ffmpeg is not available
    """
    from fastapi import HTTPException

    if not AudioSegment.converter:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is not installed. Please install ffmpeg to process audio files."
        )

    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.replace(os.path.splitext(file_path)[1], ".wav")
    audio.export(
        wav_path,
        format="wav",
        parameters=["-ar", "16000", "-ac", "1"]
    )
    return wav_path


def transcribe_with_whisper(
    file_path: str,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
    task: str = "transcribe"
) -> TranscriptionResult:
    """
    Transcribe audio file using Whisper model

    Args:
        file_path: Path to audio file
        language: Language code for transcription
        model_size: Whisper model size to use
        task: Task type (transcribe or translate)

    Returns:
        TranscriptionResult with text, language, and segments
    """
    model = load_whisper_model(model_size)
    audio = whisper.load_audio(file_path)

    result = whisper.transcribe(
        model,
        audio,
        language=language,
        task=task,
        fp16=False
    )

    # Convert segments to Pydantic models
    segments = [
        TranscriptionSegment(
            id=seg.get("id", i),
            start=seg.get("start", 0),
            end=seg.get("end", 0),
            text=seg.get("text", "")
        )
        for i, seg in enumerate(result.get("segments", []))
    ]

    return TranscriptionResult(
        text=result["text"],
        language=result.get("language"),
        segments=segments,
        model_type="whisper"
    )


def transcribe_with_qwen3_asr(
    file_path: str,
    language: Optional[str] = None,
    model_name: Optional[str] = None
) -> TranscriptionResult:
    """
    Transcribe audio file using Qwen3-ASR model.
    Supports 30+ languages and 22 Chinese dialects.

    Args:
        file_path: Path to audio file
        language: Language code for transcription
        model_name: Qwen3-ASR model name to use

    Returns:
        TranscriptionResult with text, language, and segments
    """
    from fastapi import HTTPException

    model = load_qwen3_asr_model(model_name)

    try:
        results = model.transcribe(
            audio=file_path,
            language=language,
        )

        if results and len(results) > 0:
            result = results[0]
            return TranscriptionResult(
                text=result.text,
                language=result.language,
                segments=[],  # Qwen3-ASR doesn't return segments in basic mode
                model_type="qwen3-asr"
            )
        else:
            return TranscriptionResult(
                text="",
                language=language or "unknown",
                segments=[],
                model_type="qwen3-asr"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Qwen3-ASR transcription failed: {str(e)}"
        )


def get_model_type(model: str) -> str:
    """
    Determine model type from model name

    Args:
        model: Model name

    Returns:
        Model type ('whisper' or 'qwen3-asr')
    """
    if model in STT_MODELS:
        return STT_MODELS[model]["type"]
    if model.startswith("whisper") or model in ["tiny", "base", "small", "medium", "large"]:
        return "whisper"
    if "qwen3" in model.lower() or "qwen" in model.lower():
        return "qwen3-asr"
    return "whisper"


def get_available_models() -> list[STTModelInfo]:
    """Get list of available STT models"""
    return [
        STTModelInfo(
            id=model_id,
            name=config["name"],
            type=config["type"],  # type: ignore
            description=config["description"]
        )
        for model_id, config in STT_MODELS.items()
    ]


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video file

    Args:
        video_path: Path to video file

    Returns:
        Path to extracted audio file

    Raises:
        HTTPException: If extraction fails
    """
    from fastapi import HTTPException

    audio_path = str(video_path) + ".wav"

    try:
        import moviepy.editor as mp
        video = mp.VideoFileClip(str(video_path))
        video.audio.write_audiofile(
            audio_path,
            codec='pcm_s16le',
            verbose=False,
            logger=None
        )
        video.close()
        return audio_path
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting audio: {str(e)}"
        )


async def process_transcription(
    file,
    language: Optional[str] = None,
    model: Optional[str] = None,
    task: str = "transcribe"
) -> tuple:
    """
    Process a file for transcription

    Args:
        file: UploadFile to transcribe
        language: Language code
        model: Model to use
        task: Task type

    Returns:
        Tuple of (transcription_id, result dict, files_to_cleanup)
    """
    from fastapi import HTTPException

    # Validate file type
    allowed_extensions = {
        '.mp3', '.wav', '.mp4', '.mov',
        '.mkv', '.flac', '.ogg', '.webm', '.m4a'
    }
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
        )

    # Determine model and model type
    model = model or f"whisper-{settings.WHISPER_MODEL}"
    model_type = get_model_type(model)

    # Validate model
    if model not in STT_MODELS and model_type == "whisper":
        if model in ["tiny", "base", "small", "medium", "large"]:
            model = f"whisper-{model}"
        elif not model.startswith("whisper-"):
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Generate unique ID
    transcription_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = settings.upload_dir / f"{transcription_id}_{file.filename}"
    files_to_cleanup = []

    try:
        # Save the file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        files_to_cleanup.append(str(file_path))

        # Determine if it's a video file
        video_extensions = {'.mp4', '.mov', '.mkv', '.webm'}
        is_video = file_extension in video_extensions

        # Extract audio if video file
        if is_video:
            audio_path = extract_audio_from_video(str(file_path))
            files_to_cleanup.append(audio_path)
        else:
            audio_path = str(file_path)

        # Convert to WAV format for processing (16kHz mono)
        if file_extension not in {'.wav'}:
            wav_path = convert_to_wav(audio_path)
            files_to_cleanup.append(wav_path)
            audio_path = wav_path

        # Perform transcription
        start_time = datetime.now()

        if model_type == "qwen3-asr":
            result = transcribe_with_qwen3_asr(
                audio_path,
                language=language,
                model_name=model
            )
        else:
            result = transcribe_with_whisper(
                audio_path,
                language=language,
                model_size=model,
                task=task
            )

        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        # Store result in cache
        transcription_data = {
            "id": transcription_id,
            "filename": file.filename,
            "result": {
                "text": result.text,
                "language": result.language,
                "segments": [
                    {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
                    for s in result.segments
                ],
                "model_type": result.model_type
            },
            "created_at": datetime.now().isoformat(),
            "is_video": is_video,
            "model_used": model,
            "model_type": model_type,
            "time_taken": round(time_taken, 2)
        }
        transcription_cache[transcription_id] = transcription_data

        return transcription_id, transcription_data, files_to_cleanup

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        for f in files_to_cleanup:
            safe_remove_file(f)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
