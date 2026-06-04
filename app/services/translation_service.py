"""
Translation service for timestamp-preserving transcript translation.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import numpy as np
import scipy.io.wavfile as wavfile
from pydub import AudioSegment, effects

from app.config import settings
from app.models.transcription import (
    TranscriptionSegment,
    TranslationLanguageInfo,
    TranslationModelInfo,
)
from app.services.tts_service import (
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    normalize_language_code as normalize_tts_language_code,
    synthesize_audio,
)


DEFAULT_TRANSLATION_MODEL = settings.TRANSLATION_MODEL
DEFAULT_TRANSLATION_SOURCE_LANGUAGE = "en"
DEFAULT_TRANSLATION_TARGET_LANGUAGE = "es"
DEFAULT_TRANSLATION_COMPUTE_TYPE = "int8"
DEFAULT_TRANSLATION_TOKENIZER = "facebook/nllb-200-distilled-600M"


class TranslationLanguage(TypedDict):
    """Static metadata for a supported translation language."""

    code: str
    name: str
    nllb_code: str
    tts_supported: bool


class TranslationRuntime(TypedDict):
    """Loaded translation backend."""

    translator: Any
    tokenizer: Any


TRANSLATION_LANGUAGES: list[TranslationLanguage] = [
    {"code": "en", "name": "English", "nllb_code": "eng_Latn", "tts_supported": True},
    {"code": "es", "name": "Spanish", "nllb_code": "spa_Latn", "tts_supported": True},
    {"code": "fr", "name": "French", "nllb_code": "fra_Latn", "tts_supported": True},
    {"code": "de", "name": "German", "nllb_code": "deu_Latn", "tts_supported": True},
    {"code": "ja", "name": "Japanese", "nllb_code": "jpn_Jpan", "tts_supported": True},
    {"code": "ko", "name": "Korean", "nllb_code": "kor_Hang", "tts_supported": True},
    {
        "code": "pt",
        "name": "Portuguese",
        "nllb_code": "por_Latn",
        "tts_supported": True,
    },
    {"code": "ru", "name": "Russian", "nllb_code": "rus_Cyrl", "tts_supported": True},
    {"code": "ar", "name": "Arabic", "nllb_code": "arb_Arab", "tts_supported": True},
    {"code": "hi", "name": "Hindi", "nllb_code": "hin_Deva", "tts_supported": True},
    {"code": "it", "name": "Italian", "nllb_code": "ita_Latn", "tts_supported": True},
    {"code": "nl", "name": "Dutch", "nllb_code": "nld_Latn", "tts_supported": True},
    {"code": "pl", "name": "Polish", "nllb_code": "pol_Latn", "tts_supported": True},
    {"code": "tr", "name": "Turkish", "nllb_code": "tur_Latn", "tts_supported": True},
    {"code": "uk", "name": "Ukrainian", "nllb_code": "ukr_Cyrl", "tts_supported": True},
    {
        "code": "vi",
        "name": "Vietnamese",
        "nllb_code": "vie_Latn",
        "tts_supported": True,
    },
]

LANGUAGE_ALIASES = {
    "eng": "en",
    "english": "en",
    "spa": "es",
    "spanish": "es",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "jpn": "ja",
    "japanese": "ja",
    "kor": "ko",
    "korean": "ko",
    "ara": "ar",
    "arabic": "ar",
    "hin": "hi",
    "hindi": "hi",
    "ita": "it",
    "italian": "it",
    "nld": "nl",
    "dut": "nl",
    "dutch": "nl",
    "pol": "pl",
    "polish": "pl",
    "por": "pt",
    "portuguese": "pt",
    "rus": "ru",
    "russian": "ru",
    "tur": "tr",
    "turkish": "tr",
    "ukr": "uk",
    "ukrainian": "uk",
    "vie": "vi",
    "vietnamese": "vi",
}

TRANSLATION_MODEL_CACHE: dict[str, TranslationRuntime] = {}


class TranslationBackendUnavailableError(RuntimeError):
    """Raised when the configured translation backend is unavailable."""


def normalize_translation_language_code(language: Optional[str]) -> str:
    """Normalize user-facing language input into a supported app language code."""
    normalized = (language or "").strip().lower().replace("_", "-")
    if not normalized:
        return DEFAULT_TRANSLATION_SOURCE_LANGUAGE
    normalized = normalized.split("-")[0]
    normalized = LANGUAGE_ALIASES.get(normalized, normalized)
    if normalized not in {item["code"] for item in TRANSLATION_LANGUAGES}:
        raise ValueError(f"Unsupported translation language: {language}")
    return normalized


def get_language_config(language: Optional[str]) -> TranslationLanguage:
    """Return metadata for a supported translation language."""
    normalized = normalize_translation_language_code(language)
    for item in TRANSLATION_LANGUAGES:
        if item["code"] == normalized:
            return item
    raise ValueError(f"Unsupported translation language: {language}")


def get_available_translation_models() -> list[TranslationModelInfo]:
    """Return available translation model metadata."""
    languages = [
        TranslationLanguageInfo(
            code=item["code"],
            name=item["name"],
            nllb_code=item["nllb_code"],
            tts_supported=item["tts_supported"],
        )
        for item in TRANSLATION_LANGUAGES
    ]
    return [
        TranslationModelInfo(
            id=DEFAULT_TRANSLATION_MODEL,
            name="NLLB-200 Distilled 600M CT2 int8",
            description=(
                "CPU-focused CTranslate2/int8 multilingual translation for "
                "timestamp-preserving transcript translation."
            ),
            device="cpu",
            compute_type=DEFAULT_TRANSLATION_COMPUTE_TYPE,
            languages=languages,
        )
    ]


def _load_translation_model(model_name: Optional[str] = None) -> TranslationRuntime:
    """Load and cache the CTranslate2 translator and tokenizer."""
    selected_model = model_name or DEFAULT_TRANSLATION_MODEL
    if selected_model in TRANSLATION_MODEL_CACHE:
        return TRANSLATION_MODEL_CACHE[selected_model]

    try:
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TranslationBackendUnavailableError(
            "Translation backend is not installed. Install it with "
            "`pip install -r requirements.txt`, restart the API, then try again."
        ) from exc

    try:
        model_path = snapshot_download(repo_id=selected_model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TRANSLATION_TOKENIZER)

        translator = ctranslate2.Translator(
            model_path,
            device="cpu",
            compute_type=DEFAULT_TRANSLATION_COMPUTE_TYPE,
        )
    except Exception as exc:
        raise TranslationBackendUnavailableError(
            f"Failed to load translation model '{selected_model}': {exc}"
        ) from exc

    TRANSLATION_MODEL_CACHE[selected_model] = {
        "translator": translator,
        "tokenizer": tokenizer,
    }
    return TRANSLATION_MODEL_CACHE[selected_model]


def _translate_texts(
    texts: list[str],
    source_language: str,
    target_language: str,
    model_name: Optional[str] = None,
) -> list[str]:
    """Translate a batch of texts with CTranslate2 while preserving ordering."""
    if not texts:
        return []

    runtime = _load_translation_model(model_name)
    translator = runtime["translator"]
    tokenizer = runtime["tokenizer"]
    source_nllb = get_language_config(source_language)["nllb_code"]
    target_nllb = get_language_config(target_language)["nllb_code"]

    tokenizer.src_lang = source_nllb
    source_batches = []
    target_prefixes = []
    text_index_by_batch = []

    for index, text in enumerate(texts):
        normalized_text = (text or "").strip()
        if not normalized_text:
            continue
        token_ids = tokenizer.encode(normalized_text)
        source_batches.append(tokenizer.convert_ids_to_tokens(token_ids))
        target_prefixes.append([target_nllb])
        text_index_by_batch.append(index)

    translated = ["" for _ in texts]
    if not source_batches:
        return translated

    results = translator.translate_batch(
        source_batches,
        target_prefix=target_prefixes,
        beam_size=1,
        batch_type="tokens",
        max_batch_size=2048,
    )

    for batch_index, result in enumerate(results):
        tokens = list(result.hypotheses[0])
        if tokens and tokens[0] == target_nllb:
            tokens = tokens[1:]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        translated[text_index_by_batch[batch_index]] = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        ).strip()

    return translated


def _build_translation_segments(
    source_segments: list[dict[str, Any]],
    translated_texts: list[str],
) -> list[dict[str, Any]]:
    """Copy timing from source segments and replace only the text."""
    segments = []
    for index, segment in enumerate(source_segments):
        segments.append(
            {
                "id": int(segment.get("id", index)),
                "start": float(segment.get("start", 0)),
                "end": float(segment.get("end", 0)),
                "text": translated_texts[index]
                if index < len(translated_texts)
                else "",
            }
        )
    return segments


def translate_transcription(
    transcription: dict[str, Any],
    target_language: str,
    source_language: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Translate and persist a transcription track."""
    result = cast(dict[str, Any], transcription.get("result") or {})
    source = normalize_translation_language_code(
        source_language or cast(Optional[str], result.get("language")) or "en"
    )
    target = normalize_translation_language_code(target_language)
    selected_model = model_name or DEFAULT_TRANSLATION_MODEL

    source_segments = [
        cast(dict[str, Any], segment)
        for segment in result.get("segments", [])
        if isinstance(segment, dict)
    ]

    if source_segments:
        translated_texts = _translate_texts(
            [str(segment.get("text", "")) for segment in source_segments],
            source_language=source,
            target_language=target,
            model_name=selected_model,
        )
        translated_segments = _build_translation_segments(
            source_segments, translated_texts
        )
        translated_text = " ".join(
            segment["text"].strip()
            for segment in translated_segments
            if segment["text"].strip()
        )
    else:
        source_text = str(result.get("text", "")).strip()
        translated_text = (
            _translate_texts([source_text], source, target, selected_model)[0]
            if source_text
            else ""
        )
        translated_segments = []

    translation = {
        "language": target,
        "source_language": source,
        "model": selected_model,
        "text": translated_text,
        "segments": translated_segments,
        "created_at": datetime.now().isoformat(),
    }

    translations = cast(
        dict[str, dict[str, Any]],
        transcription.setdefault("translations", {}),
    )
    previous = translations.get(target)
    if previous:
        for key in (
            "tts_audio_path",
            "tts_voice",
            "tts_model",
            "tts_speed",
            "tts_duration",
            "tts_sample_rate",
        ):
            if key in previous:
                translation[key] = previous[key]

    translations[target] = translation
    return translation


def get_translation_or_none(
    transcription: dict[str, Any],
    language: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return a saved translation by app language code."""
    if not language or language == "original":
        return None
    target = normalize_translation_language_code(language)
    translations = cast(
        dict[str, dict[str, Any]], transcription.get("translations") or {}
    )
    return translations.get(target)


def _audio_array_to_segment(audio_data: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert float audio samples into a pydub audio segment."""
    if audio_data.size == 0:
        return AudioSegment.silent(duration=0)

    clipped = np.clip(audio_data, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm)
    buffer.seek(0)
    return AudioSegment.from_file(buffer, format="wav")


def _fit_audio_to_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    """Fit synthesized speech into a timestamp window."""
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


def _translation_audio_path(transcription_id: str, language: str) -> Path:
    """Return the stable saved translated audio path."""
    return settings.tts_output_dir / f"{transcription_id}_translation_{language}.wav"


def generate_translation_tts_audio(
    transcription_id: str,
    transcription: dict[str, Any],
    target_language: str,
    tts_model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_TTS_VOICE,
    speed: float = 1.0,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Generate timestamp-aligned TTS audio for a saved translation."""
    target = normalize_translation_language_code(target_language)
    translation = get_translation_or_none(transcription, target)
    if not translation:
        raise ValueError(f"No saved translation for language '{target}'")

    existing_audio = translation.get("tts_audio_path")
    if existing_audio and Path(str(existing_audio)).exists() and not replace_existing:
        return translation

    segments = [
        TranscriptionSegment(
            id=int(segment["id"]),
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]),
        )
        for segment in translation.get("segments", [])
    ]
    if not segments:
        raise ValueError("Translated audio requires timestamped translated segments.")

    base_duration_ms = int(max(segment.end for segment in segments) * 1000)
    audio_track = AudioSegment.silent(duration=base_duration_ms)
    normalized_tts_language = normalize_tts_language_code(target)

    total_sample_rate = 0
    for segment in segments:
        text = segment.text.strip()
        target_ms = max(0, int((segment.end - segment.start) * 1000))
        if not text or target_ms == 0:
            continue

        audio_data, sample_rate, _duration = synthesize_audio(
            text=text,
            model_name=tts_model,
            voice=voice,
            speed=speed,
            pitch=1.0,
            language=normalized_tts_language,
        )
        total_sample_rate = sample_rate
        spoken_segment = _audio_array_to_segment(audio_data, sample_rate)
        audio_track = audio_track.overlay(
            _fit_audio_to_duration(spoken_segment, target_ms),
            position=int(segment.start * 1000),
        )

    audio_path = _translation_audio_path(transcription_id, target)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_track.export(audio_path, format="wav")

    translation["tts_audio_path"] = str(audio_path)
    translation["tts_voice"] = voice
    translation["tts_model"] = tts_model
    translation["tts_speed"] = speed
    translation["tts_duration"] = max(0.0, len(audio_track) / 1000)
    translation["tts_sample_rate"] = total_sample_rate or 44100
    return translation
