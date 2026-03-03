"""
FastAPI Application Entry Point

AI Transcription & TTS API
- STT: Whisper and Qwen3-ASR models for speech-to-text
- TTS: Qwen3-TTS and CosyVoice models for text-to-speech
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, get_settings
from app.routers import transcription, tts


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
        Configured FastAPI application
    """
    # Initialize FastAPI app
    app = FastAPI(
        title=settings.APP_NAME,
        description="""
## Features

### Speech-to-Text (STT)
- **Whisper models**: whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large
- **Qwen3-ASR models**: qwen3-asr-0.6b, qwen3-asr-1.7b
  - Supports 30+ languages and 22 Chinese dialects
  - Better for Chinese and Asian languages

### Text-to-Speech (TTS)
- **Qwen3-TTS models**: qwen3-tts-0.6b, qwen3-tts-1.8b, qwen3-tts-4b
- **CosyVoice models**: cosyvoice-300m, cosyvoice-300m-sft, cosyvoice-300m-instruct

### Subtitle Generation
- Generate SRT and VTT subtitle files from transcriptions
- Download subtitles for your audio/video files
        """,
        version=settings.APP_VERSION,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # Include routers
    app.include_router(transcription.router)
    app.include_router(tts.router)

    # Mount static files for frontend (in production)
    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """
        Root endpoint - serves the frontend application or API info
        """
        # Check if frontend exists
        index_html = Path(__file__).parent.parent / "frontend" / "dist" / "index.html"

        if index_html.exists():
            with open(index_html, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

        # Return API info if no frontend
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health"
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "device": settings.DEVICE,
            "app": {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION
            },
            "stt": {
                "default_whisper": settings.WHISPER_MODEL,
                "default_qwen3_asr": settings.QWEN3_ASR_MODEL,
                "available_models": [
                    "whisper-tiny", "whisper-base", "whisper-small",
                    "whisper-medium", "whisper-large",
                    "qwen3-asr-0.6b", "qwen3-asr-1.7b"
                ]
            },
            "tts": {
                "available_models": [
                    "qwen3-tts-0.6b", "qwen3-tts-1.8b", "qwen3-tts-4b",
                    "cosyvoice-300m", "cosyvoice-300m-sft", "cosyvoice-300m-instruct"
                ]
            }
        }

    return app


# Create application instance
app = create_app()


def main():
    """Run the application with uvicorn"""
    import uvicorn

    print(f"Starting {settings.APP_NAME}...")
    print(f"Version: {settings.APP_VERSION}")
    print(f"Device: {settings.DEVICE}")
    print(f"")
    print(f"STT Models:")
    print(f"  Whisper (default): {settings.WHISPER_MODEL}")
    print(f"  Qwen3-ASR (default): {settings.QWEN3_ASR_MODEL}")
    print(f"")
    print(f"TTS Models:")
    print(f"  qwen3-tts-0.6b, qwen3-tts-1.8b, qwen3-tts-4b")
    print(f"  cosyvoice-300m, cosyvoice-300m-sft, cosyvoice-300m-instruct")
    print(f"")
    print(f"Upload directory: {settings.upload_dir}")
    print(f"API available at: http://localhost:{settings.PORT}")
    print(f"Docs available at: http://localhost:{settings.PORT}/docs")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )


if __name__ == "__main__":
    main()
