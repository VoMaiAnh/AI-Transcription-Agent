"""
Application configuration using pydantic-settings
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application settings
    APP_NAME: str = "AI Transcription & TTS API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # STT Model settings
    STT_MODEL: str = "base"
    WHISPER_MODEL: str = "base"
    QWEN3_ASR_MODEL: str = "Qwen/Qwen3-ASR-1.7B"

    # TTS Model settings
    TTS_MODEL: str = "qwen3-tts-1.8b"

    # Device settings
    DEVICE: str = "cpu"  # Will be overridden by torch.cuda.is_available()

    # File settings
    MAX_FILE_SIZE: int = 52428800  # 50MB in bytes
    UPLOAD_DIR: Path = Path("./uploads")

    # CORS settings
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    @property
    def upload_dir(self) -> Path:
        """Get upload directory, creating it if it doesn't exist"""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        return self.UPLOAD_DIR

    @property
    def tts_output_dir(self) -> Path:
        """Get TTS output directory"""
        tts_dir = self.upload_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        return tts_dir


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings
