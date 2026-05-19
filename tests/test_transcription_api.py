import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
)
sys.modules.setdefault("whisper", types.SimpleNamespace())

from app.routers import transcription as transcription_router  # noqa: E402
from app.config import settings  # noqa: E402


@pytest.fixture()
def client():
    transcription_router.transcription_cache.clear()
    app = FastAPI()
    app.include_router(transcription_router.router)
    return TestClient(app)


def _cache_transcription(transcription_id: str = "tx1", is_video: bool = False):
    transcription_router.transcription_cache[transcription_id] = {
        "id": transcription_id,
        "filename": "sample.mp4" if is_video else "sample.wav",
        "result": {
            "text": "Hello world",
            "language": "en",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}],
            "model_type": "whisper",
        },
        "created_at": "2026-05-19T00:00:00",
        "is_video": is_video,
        "source_path": "sample.mp4",
        "source_size": 1024,
        "subtitle_paths": {},
        "media_paths": {},
        "model_used": "whisper-base",
        "model_type": "whisper",
        "time_taken": 1.23,
    }


def test_transcribe_contract_uses_mocked_service(client, monkeypatch):
    async def fake_process_transcription(file, language=None, model=None, task="transcribe"):
        return (
            "tx1",
            {
                "id": "tx1",
                "filename": "sample.wav",
                "result": {
                    "text": "Hello world",
                    "language": language,
                    "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}],
                    "model_type": "whisper",
                },
                "is_video": False,
                "model_used": model or "whisper-base",
                "model_type": "whisper",
                "time_taken": 0.5,
            },
            [],
        )

    monkeypatch.setattr(transcription_router, "process_transcription", fake_process_transcription)

    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("sample.wav", b"audio", "audio/wav")},
        data={"language": "en", "model": "whisper-base"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "transcription_id": "tx1",
        "filename": "sample.wav",
        "language": "en",
        "text": "Hello world",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}],
        "time_taken": 0.5,
        "model_used": "whisper-base",
        "model_type": "whisper",
        "is_video": False,
    }


def test_get_subtitle_uses_query_parameter_and_stores_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    _cache_transcription()

    response = client.get("/api/v1/subtitle/tx1?format=vtt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/vtt")
    assert response.text.startswith("WEBVTT")
    stored_path = Path(transcription_router.transcription_cache["tx1"]["subtitle_paths"]["vtt"])
    assert stored_path.exists()


def test_post_subtitle_remains_backward_compatible(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    _cache_transcription()

    response = client.post("/api/v1/subtitle/tx1", data={"format": "srt"})

    assert response.status_code == 200
    assert "00:00:00,000 --> 00:00:01,000" in response.text


def test_subtitle_rejects_unknown_format(client):
    _cache_transcription()

    response = client.get("/api/v1/subtitle/tx1?format=ass")

    assert response.status_code == 400
    assert "Unsupported subtitle format" in response.json()["detail"]


def test_embed_subtitle_contract_uses_mocked_media_pipeline(client, tmp_path, monkeypatch):
    _cache_transcription(is_video=True)
    output_path = tmp_path / "subtitled.mkv"
    output_path.write_bytes(b"video")

    def fake_create_subtitled_video(transcription_id, transcription, mode="soft", format="srt"):
        assert transcription_id == "tx1"
        assert mode == "soft"
        assert format == "srt"
        return output_path

    monkeypatch.setattr(transcription_router, "create_subtitled_video", fake_create_subtitled_video)

    response = client.post("/api/v1/subtitle/tx1/embed", data={"mode": "soft", "format": "srt"})

    assert response.status_code == 200
    assert response.content == b"video"
    assert response.headers["x-subtitle-mode"] == "soft"


def test_dub_video_contract_uses_mocked_media_pipeline(client, tmp_path, monkeypatch):
    _cache_transcription(is_video=True)
    output_path = tmp_path / "dubbed.mp4"
    output_path.write_bytes(b"dubbed")

    def fake_create_dubbed_video(transcription_id, transcription, **kwargs):
        assert transcription_id == "tx1"
        assert kwargs["voice"] == "default"
        assert kwargs["original_volume"] == 0.15
        return output_path

    monkeypatch.setattr(transcription_router, "create_dubbed_video", fake_create_dubbed_video)

    response = client.post("/api/v1/dub/tx1", data={"voice": "default"})

    assert response.status_code == 200
    assert response.content == b"dubbed"
