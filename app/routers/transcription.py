"""
Transcription Router
Handles audio/video transcription endpoints using Whisper or Qwen3-ASR models
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response

from app.models.transcription import (
    TranscriptionSegment,
    STTModelsResponse,
    SubtitleFormat,
)
from app.services.transcription_service import (
    STT_MODELS,
    transcription_cache,
    process_transcription,
    safe_remove_file,
    get_available_models,
)
from app.services.subtitle_service import (
    generate_subtitle,
    get_subtitle_media_type,
    get_subtitle_extension,
)


# Router instance
router = APIRouter(prefix="/api/v1", tags=["transcription"])


@router.get("/models", response_model=STTModelsResponse)
async def list_stt_models():
    """List available STT models"""
    from app.config import settings

    return STTModelsResponse(
        models=get_available_models(),
        default_model="whisper-base",
        default_whisper=settings.WHISPER_MODEL,
        default_qwen3_asr=settings.QWEN3_ASR_MODEL
    )


@router.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'zh', 'es')"),
    model: Optional[str] = Form(None, description="Model: whisper-tiny/base/small/medium/large or qwen3-asr-0.6b/1.7b"),
    task: str = Form("transcribe", description="Task: transcribe or translate (Whisper only)")
):
    """
    Transcribe an audio or video file using Whisper or Qwen3-ASR models.

    **Supported formats:** MP3, WAV, MP4, MOV, MKV, FLAC, OGG, WEBM, M4A

    **Whisper models:** whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large

    **Qwen3-ASR models:** qwen3-asr-0.6b, qwen3-asr-1.7b
    - Supports 30+ languages and 22 Chinese dialects
    - Better for Chinese and Asian languages
    """
    transcription_id, transcription_data, files_to_cleanup = await process_transcription(
        file=file,
        language=language,
        model=model,
        task=task
    )

    # Schedule files for cleanup
    for f in files_to_cleanup:
        background_tasks.add_task(safe_remove_file, f)

    result = transcription_data["result"]

    return {
        "success": True,
        "transcription_id": transcription_id,
        "filename": transcription_data["filename"],
        "language": result.get("language"),
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "time_taken": transcription_data["time_taken"],
        "model_used": transcription_data["model_used"],
        "model_type": transcription_data["model_type"]
    }


@router.get("/transcription/{transcription_id}")
async def get_transcription(transcription_id: str):
    """Get transcription result by ID"""
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")

    return transcription_cache[transcription_id]


@router.delete("/transcription/{transcription_id}")
async def delete_transcription(transcription_id: str):
    """Delete a transcription result"""
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")

    del transcription_cache[transcription_id]
    return {"message": "Transcription deleted", "id": transcription_id}


@router.get("/list")
async def list_transcriptions():
    """List all transcriptions in cache"""
    return {
        "transcriptions": list(transcription_cache.values()),
        "total": len(transcription_cache)
    }


@router.get("/subtitle/{transcription_id}")
async def get_subtitle(
    transcription_id: str,
    format: str = Form("srt", description="Subtitle format: srt or vtt")
):
    """
    Download subtitle file for a transcription

    Args:
        transcription_id: ID of the transcription
        format: Subtitle format (srt or vtt)
    """
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")

    transcription = transcription_cache[transcription_id]
    segments_data = transcription["result"].get("segments", [])

    if not segments_data:
        raise HTTPException(
            status_code=400,
            detail="No segments available for subtitle generation. "
                   "This model doesn't provide timing information."
        )

    # Convert to segment objects
    segments = [
        TranscriptionSegment(
            id=seg["id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"]
        )
        for seg in segments_data
    ]

    # Generate subtitle content
    subtitle_content = generate_subtitle(segments, format)

    # Get filename
    original_filename = transcription["filename"]
    base_name = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    extension = get_subtitle_extension(format)

    return Response(
        content=subtitle_content,
        media_type=get_subtitle_media_type(format),
        headers={
            "Content-Disposition": f"attachment; filename={base_name}{extension}",
            "X-Transcription-ID": transcription_id
        }
    )
