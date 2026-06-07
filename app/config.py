"""
Application configuration using pydantic-settings
"""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
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
    PARAKEET_MODEL: str = "nvidia/parakeet-tdt-0.6b-v3"

    # TTS Model settings
    TTS_MODEL: str = "Supertone/supertonic-3"

    # Translation Model settings
    TRANSLATION_MODEL: str = "JustFrederik/nllb-200-distilled-600M-ct2-int8"

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

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value: object) -> object:
        """Accept common deployment words for DEBUG in addition to booleans."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes"}:
                return True
        return value

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

    @property
    def source_media_dir(self) -> Path:
        """Get directory for retained uploaded source media"""
        source_dir = self.upload_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir

    @property
    def subtitle_output_dir(self) -> Path:
        """Get directory for generated subtitle files"""
        subtitle_dir = self.upload_dir / "subtitles"
        subtitle_dir.mkdir(parents=True, exist_ok=True)
        return subtitle_dir

    @property
    def media_output_dir(self) -> Path:
        """Get directory for generated video outputs"""
        media_dir = self.upload_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        return media_dir


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings
