"""
Subtitle generation service for creating SRT and VTT files
"""

from pathlib import Path
from typing import Any, Optional, cast

from app.config import settings
from app.models.transcription import TranscriptionSegment
from app.utils.file_utils import sanitize_filename

SUPPORTED_SUBTITLE_FORMATS = {"srt", "vtt"}
ORIGINAL_TRACK = "original"


def normalize_subtitle_format(format: str = "srt") -> str:
    """
    Validate and normalize a subtitle format string.

    Args:
        format: Subtitle format

    Returns:
        Normalized subtitle format

    Raises:
        ValueError: If format is not supported
    """
    format_lower = (format or "srt").lower().strip()

    if format_lower not in SUPPORTED_SUBTITLE_FORMATS:
        raise ValueError(f"Unsupported subtitle format: {format}. Use 'srt' or 'vtt'.")

    return format_lower


def normalize_track_language(language: Optional[str] = None) -> str:
    """Normalize a subtitle/dub track selector."""
    normalized = (language or ORIGINAL_TRACK).strip().lower()
    if not normalized or normalized in {"source", "original", "default"}:
        return ORIGINAL_TRACK
    return normalized


def _subtitle_cache_key(format: str, language: Optional[str] = None) -> str:
    """Return a stable subtitle cache key."""
    track = normalize_track_language(language)
    return format if track == ORIGINAL_TRACK else f"{track}_{format}"


def get_track_result(
    transcription: dict[str, Any],
    language: Optional[str] = None,
) -> dict[str, Any]:
    """Return the original or translated transcript result for a track."""
    track = normalize_track_language(language)
    if track == ORIGINAL_TRACK:
        return cast(dict[str, Any], transcription["result"])

    translations = cast(
        dict[str, dict[str, Any]], transcription.get("translations") or {}
    )
    translation = translations.get(track)
    if not translation:
        raise ValueError(f"No saved translation for language '{track}'.")
    return translation


def _split_milliseconds(seconds: float) -> tuple[int, int, int, int]:
    """Split seconds into timestamp parts, rounded to the nearest millisecond."""
    total_millis = max(0, round(seconds * 1000))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return hours, minutes, secs, millis


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours, minutes, secs, millis = _split_milliseconds(seconds)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Convert seconds to VTT timestamp format: HH:MM:SS.mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours, minutes, secs, millis = _split_milliseconds(seconds)

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
    segments: list["TranscriptionSegment"], format: str = "srt"
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
    format_lower = normalize_subtitle_format(format)

    if format_lower == "srt":
        return generate_srt(segments)
    if format_lower == "vtt":
        return generate_vtt(segments)

    raise ValueError(f"Unsupported subtitle format: {format}. Use 'srt' or 'vtt'.")


def get_subtitle_media_type(format: str = "srt") -> str:
    """
    Get the MIME media type for subtitle format.

    Args:
        format: Subtitle format ('srt' or 'vtt')

    Returns:
        MIME type string
    """
    format_lower = normalize_subtitle_format(format)

    if format_lower == "srt":
        return "application/x-subrip"
    if format_lower == "vtt":
        return "text/vtt"

    return "text/plain"


def get_subtitle_extension(format: str = "srt") -> str:
    """
    Get the file extension for subtitle format.

    Args:
        format: Subtitle format ('srt' or 'vtt')

    Returns:
        File extension with dot (e.g., '.srt')
    """
    format_lower = normalize_subtitle_format(format)

    if format_lower == "srt":
        return ".srt"
    if format_lower == "vtt":
        return ".vtt"

    return ".txt"


def get_segments_from_transcription(
    transcription: dict[str, Any],
    language: Optional[str] = None,
) -> list[TranscriptionSegment]:
    """
    Build typed subtitle segments from a cached transcription entry.

    Args:
        transcription: Cached transcription data
        language: Optional track language, or original

    Returns:
        List of TranscriptionSegment objects
    """
    track_result = get_track_result(transcription, language)
    return [
        TranscriptionSegment(
            id=seg["id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
        )
        for seg in track_result.get("segments", [])
    ]


def get_subtitle_output_path(
    transcription_id: str,
    transcription: dict[str, Any],
    format: str = "srt",
    language: Optional[str] = None,
) -> Path:
    """
    Build a stable subtitle output path for a transcription.

    Args:
        transcription_id: Transcription ID
        transcription: Cached transcription data
        format: Subtitle format
        language: Optional track language, or original

    Returns:
        Output path for the subtitle file
    """
    format_lower = normalize_subtitle_format(format)
    track = normalize_track_language(language)
    original_filename = transcription.get("filename") or "subtitle"
    safe_name = sanitize_filename(original_filename)
    base_name = Path(safe_name).stem or "subtitle"
    if track != ORIGINAL_TRACK:
        base_name = f"{base_name}_{track}"
    extension = get_subtitle_extension(format_lower)
    return settings.subtitle_output_dir / f"{transcription_id}_{base_name}{extension}"


def write_subtitle_file(
    transcription_id: str,
    transcription: dict[str, Any],
    format: str = "srt",
    language: Optional[str] = None,
) -> Path:
    """
    Generate and persist a subtitle file for a transcription.

    Args:
        transcription_id: Transcription ID
        transcription: Cached transcription data
        format: Subtitle format
        language: Optional track language, or original

    Returns:
        Path to generated subtitle file

    Raises:
        ValueError: If no timed segments are available or format is unsupported
    """
    format_lower = normalize_subtitle_format(format)
    segments = get_segments_from_transcription(transcription, language)

    if not segments:
        raise ValueError(
            "No segments available for subtitle generation. "
            "This model doesn't provide timing information."
        )

    subtitle_content = generate_subtitle(segments, format_lower)
    subtitle_path = get_subtitle_output_path(
        transcription_id, transcription, format_lower, language
    )
    subtitle_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path.write_text(subtitle_content, encoding="utf-8")

    transcription.setdefault("subtitle_paths", {})[
        _subtitle_cache_key(format_lower, language)
    ] = str(subtitle_path)

    return subtitle_path
