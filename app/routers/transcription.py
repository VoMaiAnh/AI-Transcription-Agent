"""
Transcription Router
Handles audio/video transcription endpoints using Whisper or Parakeet models
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse

from app.models.transcription import (
    STTModelsResponse,
)
from app.services.transcription_service import (
    transcription_cache,
    process_transcription,
    safe_remove_file,
    get_available_models,
    persist_transcription_cache,
)
from app.services.subtitle_service import (
    get_subtitle_media_type,
    write_subtitle_file,
)
from app.services.media_service import (
    TranslationNotConfiguredError,
    create_dubbed_video,
    create_subtitled_video,
    get_video_media_type,
)
from app.services.tts_service import DEFAULT_TTS_MODEL, DEFAULT_TTS_VOICE


# Router instance
router = APIRouter(prefix="/api/v1", tags=["transcription"])


def _get_transcription_or_404(transcription_id: str) -> dict:
    """Return a cached transcription or raise 404."""
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return transcription_cache[transcription_id]


def _remove_transcription_artifacts(transcription: dict) -> None:
    """Remove retained source/subtitle/media files for a deleted transcription."""
    paths = []
    if transcription.get("source_path"):
        paths.append(transcription["source_path"])
    paths.extend(transcription.get("subtitle_paths", {}).values())
    paths.extend(transcription.get("media_paths", {}).values())

    for path in paths:
        safe_remove_file(str(path))


@router.get("/models", response_model=STTModelsResponse)
async def list_stt_models():
    """List available STT models"""
    from app.config import settings

    return STTModelsResponse(
        models=get_available_models(),
        default_model="whisper-base",
        default_whisper=settings.WHISPER_MODEL,
        default_parakeet=getattr(settings, 'PARAKEET_MODEL', 'nvidia/parakeet-tdt-0.6b-v3')
    )


@router.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'zh', 'es')"),
    model: Optional[str] = Form(None, description="Model: whisper-tiny/base/small/medium/large or parakeet-tdt-0.6b"),
    task: str = Form("transcribe", description="Task: transcribe or translate (Whisper only)")
):
    """
    Transcribe an audio or video file using Whisper or Parakeet TDT models.

    **Supported formats:** MP3, WAV, FLAC, OGG, M4A, AAC, MP4, MOV, MKV, WEBM, AVI

    **Whisper models:** whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large
    - General purpose, good for most languages

    **Parakeet TDT models:** parakeet-tdt-0.6b
    - NVIDIA Parakeet TDT 0.6B v3
    - Supports 24+ languages (English, European, Russian, Ukrainian)
    - Precise timestamps, ideal for subtitle generation
    - CPU-optimized with ONNX Runtime
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
        "model_type": transcription_data["model_type"],
        "is_video": transcription_data["is_video"],
    }


@router.get("/transcription/{transcription_id}")
async def get_transcription(transcription_id: str):
    """Get transcription result by ID"""
    return _get_transcription_or_404(transcription_id)


@router.delete("/transcription/{transcription_id}")
async def delete_transcription(transcription_id: str):
    """Delete a transcription result"""
    transcription = _get_transcription_or_404(transcription_id)
    _remove_transcription_artifacts(transcription)
    del transcription_cache[transcription_id]
    persist_transcription_cache()
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
    format: str = Query("srt", description="Subtitle format: srt or vtt"),
):
    """
    Download subtitle file for a transcription

    Args:
        transcription_id: ID of the transcription
        format: Subtitle format (srt or vtt)
    """
    transcription = _get_transcription_or_404(transcription_id)

    try:
        subtitle_path = write_subtitle_file(transcription_id, transcription, format)
        persist_transcription_cache()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(
        path=subtitle_path,
        media_type=get_subtitle_media_type(format),
        filename=subtitle_path.name,
        headers={"X-Transcription-ID": transcription_id},
    )


@router.post("/subtitle/{transcription_id}")
async def post_subtitle(
    transcription_id: str,
    format: str = Form("srt", description="Subtitle format: srt or vtt"),
):
    """Backward-compatible subtitle download endpoint for older clients."""
    return await get_subtitle(transcription_id=transcription_id, format=format)


@router.post("/subtitle/{transcription_id}/embed")
async def embed_subtitle_in_video(
    transcription_id: str,
    mode: str = Form("soft", description="Subtitle mode: soft or hard"),
    format: str = Form("srt", description="Subtitle format: srt or vtt"),
):
    """
    Generate a video with subtitles embedded.

    Soft mode muxes a selectable subtitle stream into an MKV.
    Hard mode burns subtitles into the picture and returns an MP4.
    """
    transcription = _get_transcription_or_404(transcription_id)

    try:
        output_path = create_subtitled_video(
            transcription_id=transcription_id,
            transcription=transcription,
            mode=mode,
            format=format,
        )
        persist_transcription_cache()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=output_path,
        media_type=get_video_media_type(Path(output_path)),
        filename=Path(output_path).name,
        headers={
            "X-Transcription-ID": transcription_id,
            "X-Subtitle-Mode": mode,
        },
    )


@router.post("/dub/{transcription_id}")
async def dub_video(
    transcription_id: str,
    target_language: Optional[str] = Form(None, description="Optional target language. Use 'en' for Whisper translation."),
    tts_model: str = Form(DEFAULT_TTS_MODEL, description="TTS model to use"),
    voice: str = Form(DEFAULT_TTS_VOICE, description="Ignored for OmniVoice voice-design mode"),
    speed: float = Form(1.0, description="Speech speed (0.5-2.0)"),
    pitch: float = Form(1.0, description="Pitch adjustment (0.5-2.0)"),
    original_volume: float = Form(0.15, description="Original audio bed volume from 0.0 to 1.0"),
    whisper_model: str = Form("whisper-base", description="Whisper model for optional English translation"),
):
    """
    Generate a dubbed video by synthesizing timestamp-aligned speech per segment.
    """
    transcription = _get_transcription_or_404(transcription_id)

    if original_volume < 0 or original_volume > 1:
        raise HTTPException(status_code=400, detail="original_volume must be between 0.0 and 1.0")

    try:
        output_path = create_dubbed_video(
            transcription_id=transcription_id,
            transcription=transcription,
            target_language=target_language,
            tts_model=tts_model,
            voice=voice,
            speed=speed,
            pitch=pitch,
            original_volume=original_volume,
            whisper_model=whisper_model,
        )
        persist_transcription_cache()
    except TranslationNotConfiguredError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=output_path,
        media_type=get_video_media_type(Path(output_path)),
        filename=Path(output_path).name,
        headers={"X-Transcription-ID": transcription_id},
    )
