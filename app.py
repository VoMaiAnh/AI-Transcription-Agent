"""
FastAPI Audio/Video Transcription Application
Uses STT (Speech-to-Text) models for transcription
"""

import os
import asyncio
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import torch
import whisper
from pydub import AudioSegment
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Audio/Video Transcription API",
    description="Transcribe audio and video files using STT models",
    version="1.0.0"
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

# Upload directory
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)


def safe_remove_file(file_path, max_retries=3, delay=0.5):
    """Safely remove a file with retries (for Windows file locking issues)"""
    import time
    import os

    path = Path(file_path)
    if not path.exists():
        return True

    for attempt in range(max_retries):
        try:
            os.remove(path)
            return True
        except (PermissionError, OSError):
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                # Log the error but don't fail the request
                print(f"Warning: Could not remove file {path} after {max_retries} attempts")
                return False
    return False

# Model configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MODEL = None
MODEL_CACHE = {}

# In-memory storage for transcription results
transcription_cache = {}


def load_whisper_model(model_size: str = None):
    """Load the Whisper model with caching"""
    global MODEL, MODEL_CACHE

    model_size = model_size or WHISPER_MODEL

    # Return cached model if available
    if model_size in MODEL_CACHE:
        return MODEL_CACHE[model_size]

    # Load new model
    model = whisper.load_model(model_size, device=DEVICE)
    MODEL_CACHE[model_size] = model

    return model


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """Extract audio from video file"""
    try:
        import moviepy.editor as mp
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
        video.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting audio: {str(e)}")


def convert_to_wav(file_path: str) -> str:
    """Convert audio file to WAV format"""
    # Check if ffmpeg is available
    if not AudioSegment.converter:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is not installed. Please install ffmpeg to process audio files."
        )

    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.replace(os.path.splitext(file_path)[1], ".wav")
    audio.export(wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
    return wav_path


def transcribe_audio(file_path: str, language: Optional[str] = None, model_size: str = None, task: str = "transcribe") -> dict:
    """Transcribe audio file using Whisper model - handles files of any length"""
    model = load_whisper_model(model_size)

    # Load audio
    audio = whisper.load_audio(file_path)

    # Process audio in chunks (Whisper expects 30s chunks, but we handle full file)
    # Whisper's transcribe method handles long files automatically
    # We use the transcribe method which processes in sliding windows
    result = whisper.transcribe(
        model,
        audio,
        language=language,
        task=task,
        fp16=False
    )

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": result.get("segments", [])
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Audio/Video Transcription"}
    )


@app.post("/api/v1/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es', 'fr')"),
    model_size: Optional[str] = Form(None, description="Model size: tiny, base, small, medium, large")
) -> dict:
    """
    Transcribe an audio or video file.

    Supported formats: MP3, WAV, MP4, MOV, MKV, FLAC, OGG, WEBM
    """
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.mp4', '.mov', '.mkv', '.flac', '.ogg', '.webm', '.m4a'}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
        )

    # Generate unique ID for this transcription
    transcription_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = UPLOAD_DIR / f"{transcription_id}_{file.filename}"

    # Track all files to clean up
    files_to_cleanup = []

    try:
        # Save the file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        files_to_cleanup.append(str(file_path))

        # Determine if it's a video file
        video_extensions = {'.mp4', '.mov', '.mkv', '.webm'}
        is_video = file_extension in video_extensions

        # Extract audio if video file
        if is_video:
            audio_path = str(file_path) + ".wav"
        else:
            audio_path = str(file_path)

        # Convert to WAV format for processing
        if file_extension not in {'.wav'}:
            wav_path = convert_to_wav(audio_path)
            files_to_cleanup.append(wav_path)
            audio_path = wav_path

        # Perform transcription
        try:
            start_time = datetime.now()
            result = transcribe_audio(audio_path, language, model_size)
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()

            # Store result in cache
            transcription_cache[transcription_id] = {
                "id": transcription_id,
                "filename": file.filename,
                "result": result,
                "created_at": datetime.now().isoformat(),
                "is_video": is_video,
                "model_used": model_size or WHISPER_MODEL,
                "time_taken": round(time_taken, 2)
            }

            # Schedule all files for cleanup
            for f in files_to_cleanup:
                background_tasks.add_task(lambda p: safe_remove_file(p), f)

            return {
                "success": True,
                "transcription_id": transcription_id,
                "filename": file.filename,
                "language": result["language"],
                "text": result["text"],
                "segments": result.get("segments", []),
                "time_taken": round(time_taken, 2),
                "model_used": model_size or WHISPER_MODEL
            }

        except Exception as e:
            # Clean up on error - try to remove all tracked files
            for f in files_to_cleanup:
                safe_remove_file(f)
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    except Exception as e:
        # Clean up on error - try to remove all tracked files
        for f in files_to_cleanup:
            safe_remove_file(f)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/v1/transcription/{transcription_id}")
async def get_transcription(transcription_id: str) -> dict:
    """Get transcription result by ID"""
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")

    return transcription_cache[transcription_id]


@app.delete("/api/v1/transcription/{transcription_id}")
async def delete_transcription(transcription_id: str) -> dict:
    """Delete a transcription result"""
    if transcription_id not in transcription_cache:
        raise HTTPException(status_code=404, detail="Transcription not found")

    del transcription_cache[transcription_id]
    return {"message": "Transcription deleted", "id": transcription_id}


@app.get("/api/v1/list")
async def list_transcriptions() -> dict:
    """List all transcriptions in cache"""
    return {
        "transcriptions": list(transcription_cache.values()),
        "total": len(transcription_cache)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model": WHISPER_MODEL,
        "timestamp": datetime.now().isoformat()
    }


def main():
    import uvicorn

    print(f"Starting Transcription API...")
    print(f"Device: {DEVICE}")
    print(f"Model: {WHISPER_MODEL}")
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
