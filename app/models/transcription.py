"""
Transcription models and schemas
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """A segment of transcribed text with timing"""
    id: int
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    """Result from a transcription request"""
    text: str
    language: Optional[str] = None
    segments: list[TranscriptionSegment] = []
    model_type: Literal["whisper", "qwen3-asr"] = "whisper"


class TranscriptionResponse(BaseModel):
    """API response for transcription"""
    success: bool
    transcription_id: str
    filename: str
    language: Optional[str] = None
    text: str
    segments: list[TranscriptionSegment] = []
    time_taken: Optional[float] = None
    model_used: str
    model_type: Literal["whisper", "qwen3-asr"]


class TranscriptionInfo(BaseModel):
    """Information about a transcription"""
    id: str
    filename: str
    result: TranscriptionResult
    created_at: str
    is_video: bool
    model_used: str
    model_type: Literal["whisper", "qwen3-asr"]
    time_taken: float


class STTModelInfo(BaseModel):
    """Information about an STT model"""
    id: str
    name: str
    type: Literal["whisper", "qwen3-asr"]
    description: str


class STTModelsResponse(BaseModel):
    """Response for listing STT models"""
    models: list[STTModelInfo]
    default_model: str
    default_whisper: str
    default_qwen3_asr: str


class SubtitleFormat(BaseModel):
    """Subtitle format specification"""
    format: Literal["srt", "vtt"] = "srt"
    embed_text: bool = True
