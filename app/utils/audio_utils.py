"""
Audio processing utilities
"""

import os
from pathlib import Path
from typing import Optional

from pydub import AudioSegment


def convert_audio_format(
    input_path: str,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1
) -> str:
    """
    Convert audio file to specified format

    Args:
        input_path: Path to input audio file
        output_format: Target format (wav, mp3, flac, etc.)
        sample_rate: Target sample rate in Hz
        channels: Number of channels (1 for mono, 2 for stereo)

    Returns:
        Path to converted file

    Raises:
        HTTPException: If ffmpeg is not available or conversion fails
    """
    from fastapi import HTTPException

    if not AudioSegment.converter:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is not installed. Please install ffmpeg to process audio files."
        )

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)

    output_path = input_path.replace(
        os.path.splitext(input_path)[1],
        f".{output_format}"
    )

    audio.export(
        output_path,
        format=output_format,
        parameters=["-ar", str(sample_rate), "-ac", str(channels)]
    )

    return output_path


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Extract audio track from video file

    Args:
        video_path: Path to video file
        output_path: Optional output path for audio file

    Returns:
        Path to extracted audio file
    """
    if output_path is None:
        output_path = str(video_path) + ".wav"

    try:
        import moviepy.editor as mp
        video = mp.VideoFileClip(str(video_path))
        video.audio.write_audiofile(
            str(output_path),
            codec='pcm_s16le',
            verbose=False,
            logger=None
        )
        video.close()
        return str(output_path)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting audio: {str(e)}"
        )


def is_video_file(file_path: str) -> bool:
    """
    Check if file is a video based on extension

    Args:
        file_path: Path to check

    Returns:
        True if video file, False otherwise
    """
    video_extensions = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.flv'}
    return Path(file_path).suffix.lower() in video_extensions


def is_audio_file(file_path: str) -> bool:
    """
    Check if file is an audio based on extension

    Args:
        file_path: Path to check

    Returns:
        True if audio file, False otherwise
    """
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions
