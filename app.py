"""
FastAPI Audio/Video Transcription & TTS Application
- STT: Whisper and Qwen3-ASR models for speech-to-text
- TTS: Qwen3-TTS and CosyVoice models for text-to-speech
"""

import os
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import torch
from dotenv import load_dotenv

# Import routers
from routers import transcription_router, tts_router

# Load environment variables
load_dotenv()

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
QWEN3_ASR_MODEL = os.getenv("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Audio/Video Transcription & TTS API",
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
    """,
    version="2.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Include routers
app.include_router(transcription_router)
app.include_router(tts_router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Audio/Video Transcription & TTS"}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "stt": {
            "default_whisper": WHISPER_MODEL,
            "default_qwen3_asr": QWEN3_ASR_MODEL,
            "available_models": [
                "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large",
                "qwen3-asr-0.6b", "qwen3-asr-1.7b"
            ]
        },
        "tts": {
            "available_models": [
                "qwen3-tts-0.6b", "qwen3-tts-1.8b", "qwen3-tts-4b",
                "cosyvoice-300m", "cosyvoice-300m-sft", "cosyvoice-300m-instruct"
            ]
        },
        "timestamp": datetime.now().isoformat()
    }


def main():
    import uvicorn

    print(f"Starting Transcription & TTS API...")
    print(f"Device: {DEVICE}")
    print(f"")
    print(f"STT Models:")
    print(f"  Whisper (default): {WHISPER_MODEL}")
    print(f"  Qwen3-ASR (default): {QWEN3_ASR_MODEL}")
    print(f"")
    print(f"TTS Models: qwen3-tts-0.6b, qwen3-tts-1.8b, qwen3-tts-4b, cosyvoice-300m, cosyvoice-300m-sft, cosyvoice-300m-instruct")
    print(f"")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"API available at: http://localhost:8000")
    print(f"Docs available at: http://localhost:8000/docs")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()