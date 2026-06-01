"""
FastAPI Application Entry Point

AI Transcription & TTS API
- STT: Whisper and Parakeet TDT models for speech-to-text
- TTS: OmniVoice for text-to-speech and voice design

python -X utf8 -m app.main
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, get_settings
from app.routers import transcription, tts
from app.services.tts_service import get_available_models as get_available_tts_models


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
- **Parakeet TDT models**: parakeet-tdt-0.6b
  - NVIDIA Parakeet TDT 0.6B v3
  - 24+ languages (English, European, Russian, Ukrainian)
  - Precise timestamps ideal for subtitle generation
  - CPU-optimized with ONNX Runtime

### Text-to-Speech (TTS)
- **OmniVoice models**: k2-fsa/OmniVoice
  - 600+ language zero-shot coverage
  - Voice design with speaker attributes such as gender, age, pitch, accent, dialect, and whisper

### Subtitle Generation
- Generate SRT and VTT subtitle files from transcriptions
- Download subtitles for your audio/video files
- Persist generated subtitle files for replay/download
- Create soft-subtitle MKV outputs or burned-in MP4 outputs with FFmpeg
- Generate first-pass dubbed videos from timestamp-aligned TTS segments
- Dedicated Subtitle Generator tab for precise timestamps
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
                "default_parakeet": getattr(settings, 'PARAKEET_MODEL', 'nvidia/parakeet-tdt-0.6b-v3'),
                "available_models": [
                    "whisper-tiny", "whisper-base", "whisper-small",
                    "whisper-medium", "whisper-large",
                    "parakeet-tdt-0.6b"
                ]
            },
            "tts": {
                "available_models": [model.id for model in get_available_tts_models()]
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
    print(f"  Parakeet TDT: {getattr(settings, 'PARAKEET_MODEL', 'nvidia/parakeet-tdt-0.6b-v3')}")
    print(f"")
    print(f"TTS Models:")
    for model in get_available_tts_models():
        print(f"  {model.id}")
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
