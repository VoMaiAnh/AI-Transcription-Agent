"""
Subtitle generation service for creating SRT and VTT files
"""

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.transcription import TranscriptionSegment


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Convert seconds to VTT timestamp format: HH:MM:SS.mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_srt(segments: list["TranscriptionSegment"]) -> str:
    """
    Generate SRT (SubRip Subtitle) format from segments.

    SRT format:
    1
    00:00:01,000 --> 00:00:04,000
    Hello, welcome to our presentation.

    2
    00:00:04,500 --> 00:00:07,000
    Today we'll discuss AI transcription.

    Args:
        segments: List of transcription segments with start, end, and text

    Returns:
        SRT formatted string
    """
    lines = []

    for i, segment in enumerate(segments, start=1):
        # Sequence number
        lines.append(str(i))

        # Timestamp range
        start_time = format_timestamp_srt(segment.start)
        end_time = format_timestamp_srt(segment.end)
        lines.append(f"{start_time} --> {end_time}")

        # Text content
        lines.append(segment.text.strip())

        # Blank line separator
        lines.append("")

    return "\n".join(lines)


def generate_vtt(segments: list["TranscriptionSegment"]) -> str:
    """
    Generate VTT (WebVTT) format from segments.

    VTT format:
    WEBVTT

    1
    00:00:01.000 --> 00:00:04.000
    Hello, welcome to our presentation.

    2
    00:00:04.500 --> 00:00:07.000
    Today we'll discuss AI transcription.

    Args:
        segments: List of transcription segments with start, end, and text

    Returns:
        VTT formatted string
    """
    lines = ["WEBVTT", ""]

    for i, segment in enumerate(segments, start=1):
        # Sequence number (optional in VTT but included for compatibility)
        lines.append(str(i))

        # Timestamp range
        start_time = format_timestamp_vtt(segment.start)
        end_time = format_timestamp_vtt(segment.end)
        lines.append(f"{start_time} --> {end_time}")

        # Text content
        lines.append(segment.text.strip())

        # Blank line separator
        lines.append("")

    return "\n".join(lines)


def generate_subtitle(
    segments: list["TranscriptionSegment"],
    format: str = "srt"
) -> str:
    """
    Generate subtitle file content in specified format.

    Args:
        segments: List of transcription segments
        format: Output format ('srt' or 'vtt')

    Returns:
        Formatted subtitle content

    Raises:
        ValueError: If format is not supported
    """
    format_lower = format.lower()

    if format_lower == "srt":
        return generate_srt(segments)
    elif format_lower == "vtt":
        return generate_vtt(segments)
    else:
        raise ValueError(f"Unsupported subtitle format: {format}. Use 'srt' or 'vtt'.")


def get_subtitle_media_type(format: str = "srt") -> str:
    """
    Get the MIME media type for subtitle format.

    Args:
        format: Subtitle format ('srt' or 'vtt')

    Returns:
        MIME type string
    """
    format_lower = format.lower()

    if format_lower == "srt":
        return "application/x-subrip"
    elif format_lower == "vtt":
        return "text/vtt"
    else:
        return "text/plain"


def get_subtitle_extension(format: str = "srt") -> str:
    """
    Get the file extension for subtitle format.

    Args:
        format: Subtitle format ('srt' or 'vtt')

    Returns:
        File extension with dot (e.g., '.srt')
    """
    format_lower = format.lower()

    if format_lower == "srt":
        return ".srt"
    elif format_lower == "vtt":
        return ".vtt"
    else:
        return ".txt"
