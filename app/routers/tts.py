"""
TTS (Text-to-Speech) Router
Handles text-to-speech synthesis using Qwen3 models
"""

from typing import Optional
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import Response

from app.models.tts import (
    TTSModelsResponse,
    TTSVoicesResponse,
)
from app.services.tts_service import (
    tts_cache,
    process_tts,
    get_available_models,
    get_available_voices,
)


# Router instance
router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


@router.get("/models", response_model=TTSModelsResponse)
async def list_tts_models():
    """List available TTS models"""
    return TTSModelsResponse(
        models=get_available_models(),
        default_model="qwen3-tts-1.8b"
    )


@router.get("/voices", response_model=TTSVoicesResponse)
async def list_voices():
    """List available voice options"""
    return TTSVoicesResponse(
        voices=get_available_voices(),
        default_voice="default"
    )


@router.post("/synthesize")
async def synthesize_tts(
    text: str = Form(..., description="Text to synthesize"),
    model: str = Form("qwen3-tts-1.8b", description="TTS model to use"),
    voice: str = Form("default", description="Voice to use"),
    speed: float = Form(1.0, description="Speech speed (0.5-2.0)"),
    pitch: float = Form(1.0, description="Pitch adjustment (0.5-2.0)"),
    language: Optional[str] = Form(None, description="Language code (auto-detected if not specified)"),
    output_format: str = Form("wav", description="Output format: wav, mp3")
):
    """
    Synthesize speech from text using Qwen3 TTS models.

    Supported models:
    - qwen3-tts-0.6b, qwen3-tts-1.8b, qwen3-tts-4b
    - cosyvoice-300m, cosyvoice-300m-sft, cosyvoice-300m-instruct
    """
    tts_id, audio_bytes, duration, sample_rate = await process_tts(
        text=text,
        model=model,
        voice=voice,
        speed=speed,
        pitch=pitch,
        language=language,
        output_format=output_format
    )

    # Return audio response
    media_type = "audio/wav" if output_format == "wav" else "audio/mpeg"

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "X-TTS-ID": tts_id,
            "X-Duration": str(duration),
            "X-Sample-Rate": str(sample_rate)
        }
    )


@router.get("/result/{tts_id}")
async def get_tts_result(tts_id: str):
    """Get TTS result metadata by ID"""
    if tts_id not in tts_cache:
        raise HTTPException(status_code=404, detail="TTS result not found")

    return tts_cache[tts_id]


@router.delete("/result/{tts_id}")
async def delete_tts_result(tts_id: str):
    """Delete a TTS result"""
    if tts_id not in tts_cache:
        raise HTTPException(status_code=404, detail="TTS result not found")

    del tts_cache[tts_id]
    return {"message": "TTS result deleted", "id": tts_id}


@router.get("/list")
async def list_tts_results():
    """List all TTS results in cache"""
    return {
        "results": [entry.model_dump() for entry in tts_cache.values()],
        "total": len(tts_cache)
    }
