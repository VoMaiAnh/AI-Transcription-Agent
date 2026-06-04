from pathlib import Path

from app.config import settings
from app.services import translation_service
from app.services.subtitle_service import write_subtitle_file


def test_translate_transcription_preserves_segment_timing(monkeypatch):
    transcription = {
        "filename": "sample.mp4",
        "result": {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {"id": 7, "start": 1.25, "end": 3.5, "text": "Hello world"},
            ],
            "model_type": "whisper",
        },
        "translations": {},
    }

    monkeypatch.setattr(
        translation_service,
        "_translate_texts",
        lambda texts, source_language, target_language, model_name=None: [
            f"{text} in Spanish" for text in texts
        ],
    )

    translation = translation_service.translate_transcription(
        transcription,
        target_language="es",
        source_language="en",
    )

    assert translation["language"] == "es"
    assert translation["segments"] == [
        {"id": 7, "start": 1.25, "end": 3.5, "text": "Hello world in Spanish"}
    ]
    assert transcription["translations"]["es"] is translation


def test_translated_subtitle_uses_saved_translation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    transcription = {
        "filename": "sample.mp4",
        "result": {
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"},
            ],
        },
        "translations": {
            "es": {
                "language": "es",
                "source_language": "en",
                "model": "test",
                "text": "Hola",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 1.0, "text": "Hola"},
                ],
                "created_at": "2026-06-04T00:00:00",
            }
        },
        "subtitle_paths": {},
    }

    subtitle_path = write_subtitle_file("tx1", transcription, "srt", language="es")

    assert subtitle_path == Path(tmp_path) / "subtitles" / "tx1_sample_es.srt"
    assert "Hola" in subtitle_path.read_text(encoding="utf-8")
    assert transcription["subtitle_paths"]["es_srt"] == str(subtitle_path)
