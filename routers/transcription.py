"""
Transcription Router
Handles audio/video transcription endpoints using Whisper or Qwen3-ASR models
"""

import os
import uuid
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks

import torch
import whisper
from pydub import AudioSegment

# Router instance
router = APIRouter(prefix="/api/v1", tags=["transcription"])

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
QWEN3_ASR_MODEL = os.getenv("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# Available STT Models
STT_MODELS = {
    # Whisper models
    "whisper-tiny": {"type": "whisper", "name": "Whisper Tiny", "description": "Fastest, lowest accuracy"},
    "whisper-base": {"type": "whisper", "name": "Whisper Base", "description": "Fast, decent accuracy"},
    "whisper-small": {"type": "whisper", "name": "Whisper Small", "description": "Balanced speed and accuracy"},
    "whisper-medium": {"type": "whisper", "name": "Whisper Medium", "description": "Good accuracy, slower"},
    "whisper-large": {"type": "whisper", "name": "Whisper Large", "description": "Best accuracy, slowest"},
    # Qwen3-ASR models
    "qwen3-asr-0.6b": {"type": "qwen3-asr", "name": "Qwen3-ASR-0.6B", "description": "Lightweight ASR, 30+ languages, streaming support"},
    "qwen3-asr-1.7b": {"type": "qwen3-asr", "name": "Qwen3-ASR-1.7B", "description": "Standard ASR, 30+ languages + 22 Chinese dialects"},
}

# Model cache
MODEL_CACHE = {}
QWEN3_ASR_CACHE = {}

# In-memory storage for transcription results
transcription_cache = {}


def safe_remove_file(file_path, max_retries=3, delay=0.5):
    """Safely remove a file with retries (for Windows file locking issues)"""
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


def load_whisper_model(model_size: str = None):
    """Load the Whisper model with caching"""
    # Extract model size from full model name or use default
    if model_size and model_size.startswith("whisper-"):
        model_size = model_size.replace("whisper-", "")
    model_size = model_size or WHISPER_MODEL
    cache_key = f"whisper-{model_size}"

    # Return cached model if available
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    # Load new model
    model = whisper.load_model(model_size, device=DEVICE)
    MODEL_CACHE[cache_key] = model

    return model


def load_qwen3_asr_model(model_name: str = None):
    """
    Load Qwen3-ASR model with caching.
    Requires: pip install qwen-asr
    """
    model_name = model_name or QWEN3_ASR_MODEL

    # Normalize model name
    if model_name in ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]:
        model_name = f"Qwen/Qwen3-ASR-{model_name.split('-')[-1].upper()}"

    cache_key = model_name

    # Return cached model if available
    if cache_key in QWEN3_ASR_CACHE:
        return QWEN3_ASR_CACHE[cache_key]

    try:
        from qwen_asr import Qwen3ASRModel

        # Load model with appropriate settings
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
    """Convert audio file to WAV format (16kHz mono)"""
    # Check if ffmpeg is available
    if not AudioSegment.converter:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is not installed. Please install ffmpeg to process audio files."
        )

    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.replace(os.path.splitext(file_path)[1], ".wav")
    audio.export(wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
    return wav_path


def transcribe_with_whisper(
    file_path: str,
    language: Optional[str] = None,
    model_size: str = None,
    task: str = "transcribe"
) -> dict:
    """Transcribe audio file using Whisper model"""
    model = load_whisper_model(model_size)

    # Load audio
    audio = whisper.load_audio(file_path)

    # Whisper's transcribe method handles long files automatically
    result = whisper.transcribe(
        model,
        audio,
        language=language,
        task=task,
        fp16=False
    )

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": result.get("segments", []),
        "model_type": "whisper"
    }


def transcribe_with_qwen3_asr(
    file_path: str,
    language: Optional[str] = None,
    model_name: str = None,
    return_timestamps: bool = False
) -> dict:
    """
    Transcribe audio file using Qwen3-ASR model.
    Supports 30+ languages and 22 Chinese dialects.
    """
    model = load_qwen3_asr_model(model_name)

    try:
        # Qwen3-ASR transcribe method
        results = model.transcribe(
            audio=file_path,
            language=language,  # None for auto-detection
        )

        if results and len(results) > 0:
            result = results[0]
            return {
                "text": result.text,
                "language": result.language,
                "segments": [],  # Qwen3-ASR doesn't return segments in basic mode
                "model_type": "qwen3-asr"
            }
        else:
            return {
                "text": "",
                "language": language or "unknown",
                "segments": [],
                "model_type": "qwen3-asr"
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Qwen3-ASR transcription failed: {str(e)}"
        )


def get_model_type(model: str) -> str:
    """Determine model type from model name"""
    if model in STT_MODELS:
        return STT_MODELS[model]["type"]
    if model.startswith("whisper") or model in ["tiny", "base", "small", "medium", "large"]:
        return "whisper"
    if "qwen3" in model.lower() or "qwen" in model.lower():
        return "qwen3-asr"
    return "whisper"  # Default to whisper


@router.get("/models")
async def list_stt_models() -> dict:
    """List available STT models"""
    return {
        "models": [
            {
                "id": model_id,
                "name": config["name"],
                "type": config["type"],
                "description": config["description"]
            }
            for model_id, config in STT_MODELS.items()
        ],
        "default_model": "whisper-base",
        "default_whisper": WHISPER_MODEL,
        "default_qwen3_asr": QWEN3_ASR_MODEL
    }


@router.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'zh', 'es')"),
    model: Optional[str] = Form(None, description="Model: whisper-tiny/base/small/medium/large or qwen3-asr-0.6b/1.7b"),
    task: str = Form("transcribe", description="Task: transcribe or translate (Whisper only)")
) -> dict:
    """
    Transcribe an audio or video file using Whisper or Qwen3-ASR models.

    **Supported formats:** MP3, WAV, MP4, MOV, MKV, FLAC, OGG, WEBM, M4A

    **Whisper models:** whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large

    **Qwen3-ASR models:** qwen3-asr-0.6b, qwen3-asr-1.7b
    - Supports 30+ languages and 22 Chinese dialects
    - Better for Chinese and Asian languages
    """
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.mp4', '.mov', '.mkv', '.flac', '.ogg', '.webm', '.m4a'}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
        )

    # Determine model and model type
    model = model or f"whisper-{WHISPER_MODEL}"
    model_type = get_model_type(model)

    # Validate model
    if model not in STT_MODELS and model_type == "whisper":
        # Allow shorthand whisper model names
        if model in ["tiny", "base", "small", "medium", "large"]:
            model = f"whisper-{model}"
        elif not model.startswith("whisper-"):
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Generate unique ID for this transcription
    transcription_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = UPLOAD_DIR / f"{transcription_id}_{file.filename}"

    # Track all files to clean up
    files_to_cleanup = []

    try:
        # Save the file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        files_to_cleanup.append(str(file_path))

        # Determine if it's a video file
        video_extensions = {'.mp4', '.mov', '.mkv', '.webm'}
        is_video = file_extension in video_extensions

        # Extract audio if video file
        if is_video:
            audio_path = str(file_path) + ".wav"
            try:
                import moviepy.editor as mp
                video = mp.VideoFileClip(str(file_path))
                video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
                video.close()
                files_to_cleanup.append(audio_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error extracting audio: {str(e)}")
        else:
            audio_path = str(file_path)

        # Convert to WAV format for processing (16kHz mono)
        if file_extension not in {'.wav'}:
            wav_path = convert_to_wav(audio_path)
            files_to_cleanup.append(wav_path)
            audio_path = wav_path

        # Perform transcription based on model type
        try:
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
            transcription_cache[transcription_id] = {
                "id": transcription_id,
                "filename": file.filename,
                "result": result,
                "created_at": datetime.now().isoformat(),
                "is_video": is_video,
                "model_used": model,
                "model_type": model_type,
                "time_taken": round(time_taken, 2)
            }

            # Schedule all files for cleanup
            for f in files_to_cleanup:
                background_tasks.add_task(lambda p: safe_remove_file(p), f)

            return {
                "success": True,
                "transcription_id": transcription_id,
                "filename": file.filename,
                "language": result["language"],
                "text": result["text"],
                "segments": result.get("segments", []),
                "time_taken": round(time_taken, 2),
                "model_used": model,
                "model_type": model_type
            }

        except HTTPException:
            raise
        except Exception as e:
            # Clean up on error
            for f in files_to_cleanup:
                safe_remove_file(f)
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        for f in files_to_cleanup:
            safe_remove_file(f)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/transcription/{transcription_id}")
async def get_transcription(transcription_id: str) -> dict:
    """Get transcription result by ID"""
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")

    return transcription_cache[transcription_id]


@router.delete("/transcription/{transcription_id}")
async def delete_transcription(transcription_id: str) -> dict:
    """Delete a transcription result"""
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")

    del transcription_cache[transcription_id]
    return {"message": "Transcription deleted", "id": transcription_id}


@router.get("/list")
async def list_transcriptions() -> dict:
    """List all transcriptions in cache"""
    return {
        "transcriptions": list(transcription_cache.values()),
        "total": len(transcription_cache)
    }