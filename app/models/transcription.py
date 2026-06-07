"""
Transcription models and schemas
"""

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
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    model_type: Literal["whisper", "parakeet"] = "whisper"


class TranslationResult(BaseModel):
    """Persisted translated transcript and optional translated audio."""

    language: str
    source_language: Optional[str] = None
    model: str
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    created_at: str
    tts_audio_path: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_model: Optional[str] = None
    tts_speed: Optional[float] = None
    tts_duration: Optional[float] = None
    tts_sample_rate: Optional[int] = None


class TranscriptionResponse(BaseModel):
    """API response for transcription"""

    success: bool
    transcription_id: str
    filename: str
    language: Optional[str] = None
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    time_taken: Optional[float] = None
    model_used: str
    model_type: Literal["whisper", "parakeet"]
    is_video: bool = False


class TranscriptionInfo(BaseModel):
    """Information about a transcription"""

    id: str
    filename: str
    result: TranscriptionResult
    created_at: str
    is_video: bool
    model_used: str
    model_type: Literal["whisper", "parakeet"]
    time_taken: float
    source_size: Optional[int] = None
    subtitle_paths: dict[str, str] = Field(default_factory=dict)
    media_paths: dict[str, str] = Field(default_factory=dict)
    translations: dict[str, TranslationResult] = Field(default_factory=dict)


class STTModelInfo(BaseModel):
    """Information about an STT model"""

    id: str
    name: str
    type: Literal["whisper", "parakeet"]
    description: str


class STTModelsResponse(BaseModel):
    """Response for listing STT models"""

    models: list[STTModelInfo]
    default_model: str
    default_whisper: str
    default_parakeet: Optional[str] = None


class TranslationLanguageInfo(BaseModel):
    """Supported translation language shown in the UI."""

    code: str
    name: str
    nllb_code: str
    tts_supported: bool = True


class TranslationModelInfo(BaseModel):
    """Information about a translation model."""

    id: str
    name: str
    description: str
    device: str
    compute_type: str
    languages: list[TranslationLanguageInfo]


class TranslationModelsResponse(BaseModel):
    """Response for listing translation models."""

    models: list[TranslationModelInfo]
    default_model: str
    default_source_language: str
    default_target_language: str


class SubtitleFormat(BaseModel):
    """Subtitle format specification"""

    format: Literal["srt", "vtt"] = "srt"
    embed_text: bool = True
