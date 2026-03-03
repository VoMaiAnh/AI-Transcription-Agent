"""
TTS (Text-to-Speech) models and schemas
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class TTSModelInfo(BaseModel):
    """Information about a TTS model"""
    id: str
    name: str
    description: str
    sample_rate: int
    languages: list[str]


class TTSModelsResponse(BaseModel):
    """Response for listing TTS models"""
    models: list[TTSModelInfo]
    default_model: str


class TTSVoiceInfo(BaseModel):
    """Information about a TTS voice"""
    id: str
    name: str
    language: str


class TTSVoicesResponse(BaseModel):
    """Response for listing TTS voices"""
    voices: list[TTSVoiceInfo]
    default_voice: str


class TTSRequest(BaseModel):
    """Request for TTS synthesis"""
    text: str = Field(..., min_length=1, max_length=5000)
    model: str = "qwen3-tts-1.8b"
    voice: str = "default"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    pitch: float = Field(1.0, ge=0.5, le=2.0)
    language: Optional[str] = None
    output_format: Literal["wav", "mp3"] = "wav"


class TTSResponse(BaseModel):
    """Response for TTS synthesis"""
    success: bool
    tts_id: str
    text: str
    model: str
    voice: str
    duration_seconds: float
    audio_url: Optional[str] = None
    created_at: str


class TTSCacheEntry(BaseModel):
    """Cached TTS result"""
    id: str
    text: str
    model: str
    voice: str
    speed: float
    pitch: float
    language: Optional[str]
    duration: float
    sample_rate: int
    created_at: str


class TTSListResponse(BaseModel):
    """Response for listing TTS results"""
    results: list[TTSCacheEntry]
    total: int
