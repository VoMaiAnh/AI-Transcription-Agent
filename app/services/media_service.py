"""
Media output service for subtitle embedding and dubbed video generation.
"""

import io
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import scipy.io.wavfile as wavfile
from pydub import AudioSegment, effects

from app.config import settings
from app.models.transcription import TranscriptionSegment
from app.services.subtitle_service import (
    ORIGINAL_TRACK,
    get_segments_from_transcription,
    normalize_subtitle_format,
    normalize_track_language,
    write_subtitle_file,
)
from app.services.tts_service import (
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    synthesize_audio,
)
from app.utils.file_utils import safe_remove_file


class TranslationNotConfiguredError(RuntimeError):
    """Raised when a requested translation target is not available."""


def get_video_media_type(path: Path) -> str:
    """Return an appropriate media type for a generated video file."""
    if path.suffix.lower() == ".mkv":
        return "video/x-matroska"
    return "video/mp4"


def ensure_ffmpeg_available() -> None:
    """Raise a helpful error if FFmpeg is unavailable."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or is not available on PATH")


def _run_ffmpeg(args: list[str]) -> None:
    """Run FFmpeg and turn stderr into a concise Python exception."""
    ensure_ffmpeg_available()
    completed = subprocess.run(
        ["ffmpeg", "-y", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg failed: {message}")


def _require_video_source(transcription: dict[str, Any]) -> Path:
    """Return the retained source video path for a transcription."""
    if not transcription.get("is_video"):
        raise ValueError(
            "Subtitle embedding and dubbing require an uploaded video source"
        )

    source_path = Path(transcription.get("source_path", ""))
    if not source_path.exists():
        raise FileNotFoundError(
            "Original video source is no longer available for this transcription"
        )

    return source_path


def _escape_subtitle_filter_path(path: Path) -> str:
    """Escape a filesystem path for FFmpeg's subtitles video filter."""
    normalized = path.resolve().as_posix()
    normalized = normalized.replace("\\", "/")
    normalized = normalized.replace(":", r"\:")
    normalized = normalized.replace("'", r"\'")
    return normalized


def create_subtitled_video(
    transcription_id: str,
    transcription: dict[str, Any],
    mode: str = "soft",
    format: str = "srt",
    language: Optional[str] = None,
) -> Path:
    """
    Create a video with generated subtitles embedded.

    Args:
        transcription_id: Transcription ID
        transcription: Cached transcription data
        mode: "soft" to mux a subtitle stream, "hard" to burn captions into video
        format: Subtitle format to generate
        language: Optional subtitle track language, or original

    Returns:
        Path to the generated video
    """
    mode_lower = (mode or "soft").lower().strip()
    if mode_lower not in {"soft", "hard"}:
        raise ValueError("Subtitle embed mode must be 'soft' or 'hard'")

    format_lower = normalize_subtitle_format(format)
    track = normalize_track_language(language)
    source_path = _require_video_source(transcription)
    subtitle_path = write_subtitle_file(
        transcription_id,
        transcription,
        format_lower,
        language=track,
    )

    output_dir = settings.media_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    media_paths = cast(dict[str, str], transcription.setdefault("media_paths", {}))
    cache_key = (
        f"subtitles_{mode_lower}_{format_lower}"
        if track == ORIGINAL_TRACK
        else f"subtitles_{track}_{mode_lower}_{format_lower}"
    )

    if mode_lower == "soft":
        output_path = (
            output_dir / f"{transcription_id}_subtitles_{format_lower}.mkv"
            if track == ORIGINAL_TRACK
            else output_dir / f"{transcription_id}_subtitles_{track}_{format_lower}.mkv"
        )
        subtitle_codec = "srt" if format_lower == "srt" else "webvtt"
        args = [
            "-i",
            str(source_path),
            "-i",
            str(subtitle_path),
            "-map",
            "0",
            "-map",
            "1:0",
            "-c",
            "copy",
            "-c:s",
            subtitle_codec,
            str(output_path),
        ]
    else:
        output_path = (
            output_dir / f"{transcription_id}_subtitles_burned.mp4"
            if track == ORIGINAL_TRACK
            else output_dir / f"{transcription_id}_subtitles_{track}_burned.mp4"
        )
        escaped_subtitle_path = _escape_subtitle_filter_path(subtitle_path)
        args = [
            "-i",
            str(source_path),
            "-vf",
            f"subtitles='{escaped_subtitle_path}'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]

    _run_ffmpeg(args)

    media_paths[cache_key] = str(output_path)
    return output_path


def _audio_array_to_segment(audio_data: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert float audio samples into a pydub segment."""
    if audio_data.size == 0:
        return AudioSegment.silent(duration=0)

    clipped = np.clip(audio_data, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm)
    buffer.seek(0)
    return AudioSegment.from_file(buffer, format="wav")


def _fit_audio_to_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    """Fit synthesized audio into a segment's timestamp window."""
    if target_ms <= 0:
        return AudioSegment.silent(duration=0)

    if len(audio) == 0:
        return AudioSegment.silent(duration=target_ms)

    if len(audio) > target_ms + 20:
        playback_speed = len(audio) / target_ms
        if 1.0 < playback_speed <= 2.0 and target_ms >= 150:
            try:
                audio = effects.speedup(
                    audio,
                    playback_speed=playback_speed,
                    chunk_size=50,
                    crossfade=10,
                )
            except Exception:
                pass
        audio = audio[:target_ms]

    if len(audio) < target_ms:
        audio += AudioSegment.silent(duration=target_ms - len(audio))

    return audio


def _with_volume(audio: AudioSegment, volume: float) -> AudioSegment:
    """Scale an audio segment by a 0.0-1.0 volume factor."""
    if volume <= 0:
        return AudioSegment.silent(duration=len(audio))
    if volume >= 1:
        return audio
    return audio + (20 * math.log10(volume))


def _segments_for_render_track(
    source_path: Path,
    transcription: dict[str, Any],
    language: Optional[str],
    target_language: Optional[str],
    whisper_model: str,
) -> list[TranscriptionSegment]:
    """
    Return original, saved translated, or legacy Whisper-English segments.

    The new workflow passes ``language`` and expects a saved translation. If old
    clients only pass ``target_language=en``, keep the previous Whisper fallback.
    """
    source_language = (transcription.get("result", {}).get("language") or "").lower()
    if language is not None:
        track = normalize_track_language(language)
        if track == ORIGINAL_TRACK:
            return get_segments_from_transcription(transcription)
        try:
            return get_segments_from_transcription(transcription, track)
        except ValueError as exc:
            raise TranslationNotConfiguredError(str(exc)) from exc

    target = (target_language or "").lower().strip()
    if not target or target in {"original", source_language}:
        return get_segments_from_transcription(transcription)

    if target != "en":
        raise TranslationNotConfiguredError(
            "Translation is currently only available for target_language='en'"
        )

    from app.services.transcription_service import transcribe_with_whisper

    translated = transcribe_with_whisper(
        str(source_path),
        language=source_language or None,
        model_size=whisper_model,
        task="translate",
    )
    return translated.segments


def _saved_translation_audio_path(
    transcription: dict[str, Any],
    language: Optional[str],
) -> Optional[Path]:
    """Return saved translated audio if the selected track has one."""
    track = normalize_track_language(language)
    if track == ORIGINAL_TRACK:
        return None
    translation = (transcription.get("translations") or {}).get(track)
    if not translation:
        return None
    audio_path = Path(str(translation.get("tts_audio_path") or ""))
    if not audio_path.exists() or not audio_path.is_file():
        return None
    return audio_path


def create_dubbed_video(
    transcription_id: str,
    transcription: dict[str, Any],
    language: Optional[str] = None,
    target_language: Optional[str] = None,
    tts_model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_TTS_VOICE,
    speed: float = 1.0,
    pitch: float = 1.0,
    original_volume: float = 0.15,
    whisper_model: str = "whisper-base",
) -> Path:
    """
    Generate a dubbed video from timestamped transcription segments.

    Args:
        transcription_id: Transcription ID
        transcription: Cached transcription data
        language: Optional saved transcript track language, or original.
        target_language: Legacy optional target language. "en" uses Whisper translate.
        tts_model: TTS model to use per segment
        voice: TTS voice
        speed: TTS speed
        pitch: TTS pitch
        original_volume: Background volume for the original audio, 0.0-1.0
        whisper_model: Whisper model for optional English translation

    Returns:
        Path to the generated dubbed video
    """
    source_path = _require_video_source(transcription)
    segments = _segments_for_render_track(
        source_path,
        transcription,
        language,
        target_language,
        whisper_model,
    )

    if not segments:
        raise ValueError("No timed segments available for dubbing")

    output_dir = settings.media_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if language is not None:
        target = normalize_track_language(language)
    else:
        target = (target_language or ORIGINAL_TRACK).lower().strip()
    if not target:
        target = ORIGINAL_TRACK
    output_path = output_dir / f"{transcription_id}_dubbed_{target}.mp4"
    audio_path = output_dir / f"{transcription_id}_dubbed_{target}.wav"

    try:
        source_audio = AudioSegment.from_file(source_path)
        base_duration_ms = max(
            len(source_audio),
            int(max(segment.end for segment in segments) * 1000),
        )
    except Exception:
        source_audio = AudioSegment.silent(duration=0)
        base_duration_ms = int(max(segment.end for segment in segments) * 1000)

    saved_audio_path = _saved_translation_audio_path(transcription, language)
    if saved_audio_path:
        dubbed_track = AudioSegment.from_file(saved_audio_path)
        if len(dubbed_track) < base_duration_ms:
            dubbed_track += AudioSegment.silent(
                duration=base_duration_ms - len(dubbed_track)
            )
        elif len(dubbed_track) > base_duration_ms:
            base_duration_ms = len(dubbed_track)
    else:
        dubbed_track = AudioSegment.silent(duration=base_duration_ms)
        tts_language = (
            target
            if target != ORIGINAL_TRACK
            else transcription.get("result", {}).get("language")
        )

        for segment in segments:
            target_ms = max(0, int((segment.end - segment.start) * 1000))
            if target_ms == 0:
                continue

            audio_data, sample_rate, _duration = synthesize_audio(
                text=segment.text.strip(),
                model_name=tts_model,
                voice=voice,
                speed=speed,
                pitch=pitch,
                language=tts_language,
            )
            spoken_segment = _audio_array_to_segment(audio_data, sample_rate)
            spoken_segment = _fit_audio_to_duration(spoken_segment, target_ms)
            dubbed_track = dubbed_track.overlay(
                spoken_segment, position=int(segment.start * 1000)
            )

    if original_volume > 0 and len(source_audio) > 0:
        source_bed = source_audio[:base_duration_ms]
        if len(source_bed) < base_duration_ms:
            source_bed += AudioSegment.silent(
                duration=base_duration_ms - len(source_bed)
            )
        mixed_audio = _with_volume(source_bed, min(original_volume, 1.0)).overlay(
            dubbed_track
        )
    else:
        mixed_audio = dubbed_track

    mixed_audio.export(audio_path, format="wav")

    _run_ffmpeg(
        [
            "-i",
            str(source_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    )

    safe_remove_file(str(audio_path))
    media_paths = cast(dict[str, str], transcription.setdefault("media_paths", {}))
    media_paths[f"dubbed_{target}"] = str(output_path)
    return output_path


def create_dubbed_subtitled_video(
    transcription_id: str,
    transcription: dict[str, Any],
    language: Optional[str] = None,
    subtitle_mode: str = "hard",
    subtitle_format: str = "srt",
    target_language: Optional[str] = None,
    tts_model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_TTS_VOICE,
    speed: float = 1.0,
    pitch: float = 1.0,
    original_volume: float = 0.15,
    whisper_model: str = "whisper-base",
) -> Path:
    """Create one final video containing both the selected dub and subtitle track."""
    mode_lower = (subtitle_mode or "hard").lower().strip()
    if mode_lower not in {"soft", "hard"}:
        raise ValueError("Subtitle mode must be 'soft' or 'hard'")

    format_lower = normalize_subtitle_format(subtitle_format)
    track = normalize_track_language(language)

    dubbed_path = create_dubbed_video(
        transcription_id=transcription_id,
        transcription=transcription,
        language=track,
        target_language=target_language,
        tts_model=tts_model,
        voice=voice,
        speed=speed,
        pitch=pitch,
        original_volume=original_volume,
        whisper_model=whisper_model,
    )
    subtitle_path = write_subtitle_file(
        transcription_id,
        transcription,
        format_lower,
        language=track,
    )

    output_dir = settings.media_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    media_paths = cast(dict[str, str], transcription.setdefault("media_paths", {}))
    cache_key = f"combined_{track}_{mode_lower}_{format_lower}"

    if mode_lower == "soft":
        output_path = (
            output_dir
            / f"{transcription_id}_dubbed_{track}_subtitles_{format_lower}.mkv"
        )
        subtitle_codec = "srt" if format_lower == "srt" else "webvtt"
        args = [
            "-i",
            str(dubbed_path),
            "-i",
            str(subtitle_path),
            "-map",
            "0",
            "-map",
            "1:0",
            "-c",
            "copy",
            "-c:s",
            subtitle_codec,
            str(output_path),
        ]
    else:
        output_path = output_dir / f"{transcription_id}_dubbed_{track}_burned.mp4"
        escaped_subtitle_path = _escape_subtitle_filter_path(subtitle_path)
        args = [
            "-i",
            str(dubbed_path),
            "-vf",
            f"subtitles='{escaped_subtitle_path}'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]

    _run_ffmpeg(args)

    media_paths[cache_key] = str(output_path)
    return output_path
