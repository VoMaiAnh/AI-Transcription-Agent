from pathlib import Path

import pytest

from app.config import settings
from app.models.transcription import TranscriptionSegment
from app.services.subtitle_service import (
    format_timestamp_srt,
    format_timestamp_vtt,
    generate_srt,
    generate_vtt,
    normalize_subtitle_format,
    write_subtitle_file,
)


def test_timestamp_formatting_rounds_to_nearest_millisecond():
    assert format_timestamp_srt(3661.9996) == "01:01:02,000"
    assert format_timestamp_vtt(61.2345) == "00:01:01.234"
    assert format_timestamp_srt(-2.0) == "00:00:00,000"


def test_generate_srt_and_vtt_from_segments():
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=1.5, text=" Hello "),
        TranscriptionSegment(id=1, start=2.0, end=3.25, text="World"),
    ]

    assert generate_srt(segments) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Hello\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,250\n"
        "World\n"
    )
    assert generate_vtt(segments).startswith("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.500")


def test_normalize_subtitle_format_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported subtitle format"):
        normalize_subtitle_format("ass")


def test_write_subtitle_file_persists_output_and_updates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    transcription = {
        "filename": "unsafe name!.mp4",
        "result": {
            "segments": [
                {"id": 0, "start": 0, "end": 1, "text": "Hello there"},
            ],
        },
        "subtitle_paths": {},
    }

    subtitle_path = write_subtitle_file("abc123", transcription, "srt")

    assert subtitle_path == Path(tmp_path) / "subtitles" / "abc123_unsafe_name.srt"
    assert subtitle_path.read_text(encoding="utf-8").endswith("Hello there\n")
    assert transcription["subtitle_paths"]["srt"] == str(subtitle_path)
