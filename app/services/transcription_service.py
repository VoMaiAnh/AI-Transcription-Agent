"""
Transcription service for audio/video to text conversion
"""

import json
import os
import re
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
from app.utils.file_utils import sanitize_filename


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
    # Parakeet TDT models (CPU-optimized)
    "parakeet-tdt-0.6b": {
        "type": "parakeet",
        "name": "Parakeet TDT 0.6B v3",
        "description": "NVIDIA Parakeet, 24+ languages, precise timestamps, CPU-friendly"
    },
}

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model caches
MODEL_CACHE = {}
PARAKEET_MODEL_CACHE = {}

# In-memory storage for transcription results
transcription_cache = {}
TRANSCRIPTION_INDEX_PATH = settings.upload_dir / "transcriptions_index.json"

ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.avi'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
UPLOAD_CHUNK_SIZE = 1024 * 1024
UUID_PREFIX_RE = re.compile(
    r"^(?P<id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_(?P<filename>.+)$"
)


def _parse_subtitle_timestamp(value: str) -> float:
    """Parse SRT/VTT timestamp text into seconds."""
    timestamp = value.strip().replace(",", ".")
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_subtitle_segments(path: Path) -> list[dict]:
    """Best-effort parser for saved SRT/VTT files."""
    if not path.exists():
        return []

    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="ignore").strip())
    segments = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].upper() == "WEBVTT":
            continue

        timestamp_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timestamp_index < 0:
            continue

        start_text, end_text = lines[timestamp_index].split("-->", 1)
        end_text = end_text.split()[0]
        text = " ".join(lines[timestamp_index + 1:]).strip()

        if not text:
            continue

        try:
            segments.append({
                "id": len(segments),
                "start": _parse_subtitle_timestamp(start_text),
                "end": _parse_subtitle_timestamp(end_text),
                "text": text,
            })
        except (ValueError, IndexError):
            continue

    return segments


def persist_transcription_cache() -> None:
    """Persist transcription metadata so Archive survives backend restarts."""
    TRANSCRIPTION_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = TRANSCRIPTION_INDEX_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(list(transcription_cache.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(TRANSCRIPTION_INDEX_PATH)


def _recover_transcription_from_source(source_path: Path) -> Optional[dict]:
    """Rebuild a basic transcription record from retained media and subtitle files."""
    match = UUID_PREFIX_RE.match(source_path.name)
    if not match:
        return None

    transcription_id = match.group("id")
    filename = match.group("filename")
    file_extension = source_path.suffix.lower()
    is_video = file_extension in ALLOWED_VIDEO_EXTENSIONS

    subtitle_paths = {}
    segments = []
    for subtitle_path in sorted(settings.subtitle_output_dir.glob(f"{transcription_id}_*")):
        subtitle_extension = subtitle_path.suffix.lower().lstrip(".")
        if subtitle_extension in {"srt", "vtt"}:
            subtitle_paths[subtitle_extension] = str(subtitle_path)
            if not segments:
                segments = _parse_subtitle_segments(subtitle_path)

    media_paths = {
        media_path.stem.replace(f"{transcription_id}_", "", 1): str(media_path)
        for media_path in settings.media_output_dir.glob(f"{transcription_id}_*")
        if media_path.is_file()
    }

    return {
        "id": transcription_id,
        "filename": filename,
        "result": {
            "text": " ".join(segment["text"] for segment in segments),
            "language": None,
            "segments": segments,
            "model_type": "whisper",
        },
        "created_at": datetime.fromtimestamp(source_path.stat().st_mtime).isoformat(),
        "is_video": is_video,
        "source_path": str(source_path),
        "source_size": source_path.stat().st_size,
        "subtitle_paths": subtitle_paths,
        "media_paths": media_paths,
        "model_used": "Recovered from uploads",
        "model_type": "whisper",
        "time_taken": 0,
    }


def load_transcription_cache() -> None:
    """Load persisted metadata, falling back to recovered upload files."""
    transcription_cache.clear()

    if TRANSCRIPTION_INDEX_PATH.exists():
        try:
            entries = json.loads(TRANSCRIPTION_INDEX_PATH.read_text(encoding="utf-8"))
            for entry in entries:
                if isinstance(entry, dict) and entry.get("id"):
                    transcription_cache[entry["id"]] = entry
        except (OSError, json.JSONDecodeError):
            transcription_cache.clear()

    recovered_any = False
    for source_path in settings.source_media_dir.iterdir():
        if not source_path.is_file():
            continue
        recovered = _recover_transcription_from_source(source_path)
        if recovered and recovered["id"] not in transcription_cache:
            transcription_cache[recovered["id"]] = recovered
            recovered_any = True

    if recovered_any or not TRANSCRIPTION_INDEX_PATH.exists():
        persist_transcription_cache()


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


def load_parakeet_model(model_name: Optional[str] = None):
    """
    Load Parakeet TDT model with caching.
    CPU-optimized using ONNX Runtime.

    Args:
        model_name: Name of the Parakeet model to load

    Returns:
        Loaded Parakeet model/pipeline

    Raises:
        HTTPException: If model loading fails
    """
    from fastapi import HTTPException

    model_name = model_name or "nvidia/parakeet-tdt-0.6b-v3"

    # Normalize model name
    if model_name in ["parakeet-tdt-0.6b", "parakeet-tdt-0.6b-v3"]:
        model_name = "nvidia/parakeet-tdt-0.6b-v3"

    cache_key = model_name

    if cache_key in PARAKEET_MODEL_CACHE:
        return PARAKEET_MODEL_CACHE[cache_key]

    try:
        from app.services.parakeet_service import load_parakeet_model as load_model

        model = load_model(model_name)
        PARAKEET_MODEL_CACHE[cache_key] = model
        return model

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Parakeet model not installed. Install with: pip install optimum[onnxruntime] transformers torch soundfile scipy"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load Parakeet model: {str(e)}"
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


def transcribe_with_parakeet(
    file_path: str,
    language: Optional[str] = None,
    model_name: Optional[str] = None
) -> TranscriptionResult:
    """
    Transcribe audio file using Parakeet TDT model.
    CPU-optimized with chunked inference for long audio.

    Args:
        file_path: Path to audio file
        language: Language code for transcription
        model_name: Parakeet model name

    Returns:
        TranscriptionResult with text, language, and segments
    """
    from app.services.parakeet_service import transcribe_with_parakeet as transcribe

    return transcribe(
        audio_path=file_path,
        language=language,
        model_name=model_name,
        chunk_length=30.0,  # 30 second chunks
        return_timestamps=True
    )


def get_model_type(model: str) -> str:
    """
    Determine model type from model name

    Args:
        model: Model name

    Returns:
        Model type ('whisper' or 'parakeet')
    """
    if model in STT_MODELS:
        return STT_MODELS[model]["type"]
    if model.startswith("whisper") or model in ["tiny", "base", "small", "medium", "large"]:
        return "whisper"
    if "parakeet" in model.lower():
        return "parakeet"
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


def format_supported_extensions() -> str:
    """Return supported extensions as a stable, readable string."""
    return ", ".join(sorted(ALLOWED_EXTENSIONS))


async def save_upload_file(file, destination: Path) -> int:
    """
    Stream an uploaded file to disk while enforcing the configured size limit.

    Args:
        file: UploadFile to save
        destination: Final destination path

    Returns:
        Number of bytes written
    """
    from fastapi import HTTPException

    total_size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "wb") as output:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > settings.MAX_FILE_SIZE:
                output.close()
                safe_remove_file(str(destination))
                max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {max_mb:.0f} MB."
                )

            output.write(chunk)

    if total_size == 0:
        safe_remove_file(str(destination))
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return total_size


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
    safe_filename = sanitize_filename(getattr(file, "filename", None))
    file_extension = Path(safe_filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: {format_supported_extensions()}"
        )

    # Determine model and model type
    model = model or f"whisper-{settings.WHISPER_MODEL}"
    model_type = get_model_type(model)

    if task not in {"transcribe", "translate"}:
        raise HTTPException(status_code=400, detail="Task must be 'transcribe' or 'translate'")

    if task == "translate" and model_type != "whisper":
        raise HTTPException(status_code=400, detail="Translate task is only supported by Whisper models")

    # Validate model
    if model not in STT_MODELS and model_type == "whisper":
        if model in ["tiny", "base", "small", "medium", "large"]:
            model = f"whisper-{model}"
        elif not model.startswith("whisper-"):
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Generate unique ID
    transcription_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = settings.source_media_dir / f"{transcription_id}_{safe_filename}"
    files_to_cleanup = []
    saved_size = 0

    try:
        # Save the original source for later subtitle embedding/dubbing workflows.
        saved_size = await save_upload_file(file, file_path)

        # Determine if it's a video file
        is_video = file_extension in ALLOWED_VIDEO_EXTENSIONS

        # Extract audio if video file
        if is_video:
            audio_path = extract_audio_from_video(str(file_path))
            files_to_cleanup.append(audio_path)
        else:
            audio_path = str(file_path)

        # Convert source audio to WAV format for processing (16kHz mono).
        # Video extraction already writes a WAV audio track.
        if not is_video and file_extension != '.wav':
            wav_path = convert_to_wav(audio_path)
            if wav_path != audio_path:
                files_to_cleanup.append(wav_path)
            audio_path = wav_path

        # Perform transcription
        start_time = datetime.now()

        if model_type == "parakeet":
            result = transcribe_with_parakeet(
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
            "filename": safe_filename,
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
            "source_path": str(file_path),
            "source_size": saved_size,
            "subtitle_paths": {},
            "media_paths": {},
            "model_used": model,
            "model_type": model_type,
            "time_taken": round(time_taken, 2)
        }
        transcription_cache[transcription_id] = transcription_data
        persist_transcription_cache()

        return transcription_id, transcription_data, files_to_cleanup

    except HTTPException:
        for f in files_to_cleanup:
            safe_remove_file(f)
        safe_remove_file(str(file_path))
        raise
    except Exception as e:
        # Clean up on error
        for f in files_to_cleanup:
            safe_remove_file(f)
        safe_remove_file(str(file_path))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


load_transcription_cache()
