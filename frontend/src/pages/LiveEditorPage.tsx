import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import * as api from "../api/client";
import { TranscriptionResponse, TTSModel, TTSVoice } from "../types";
import {
  ALLOWED_MEDIA_EXTENSIONS,
  downloadBlob,
  formatFileSize,
  formatTime,
  validateMediaFile,
} from "./pageUtils";

const FALLBACK_TTS_MODELS: TTSModel[] = [
  {
    id: "Supertone/supertonic-3",
    name: "Supertonic 3",
    description:
      "Lightning-fast on-device multilingual TTS using ONNX Runtime.",
    sample_rate: 44100,
    languages: [
      "en",
      "ko",
      "ja",
      "ar",
      "bg",
      "cs",
      "da",
      "de",
      "el",
      "es",
      "et",
      "fi",
      "fr",
      "hi",
      "hr",
      "hu",
      "id",
      "it",
      "lt",
      "lv",
      "nl",
      "pl",
      "pt",
      "ro",
      "ru",
      "sk",
      "sl",
      "sv",
      "tr",
      "uk",
      "vi",
      "na",
    ],
    model_family: "supertonic",
    supports_instructions: false,
    supports_voice_presets: true,
    requires_reference_audio: false,
    features: [
      "31 language codes plus unknown-language fallback",
      "Built-in voice styles M1-M5 and F1-F5",
      "ONNX Runtime local inference",
      "Expression tags such as <laugh>, <breath>, and <sigh>",
    ],
  },
];

const FALLBACK_TTS_VOICES: TTSVoice[] = [
  {
    id: "M1",
    name: "Supertonic M1",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "M2",
    name: "Supertonic M2",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "M3",
    name: "Supertonic M3",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "M4",
    name: "Supertonic M4",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "M5",
    name: "Supertonic M5",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "F1",
    name: "Supertonic F1",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "F2",
    name: "Supertonic F2",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "F3",
    name: "Supertonic F3",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "F4",
    name: "Supertonic F4",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
  {
    id: "F5",
    name: "Supertonic F5",
    language: "multilingual",
    model_family: "supertonic",
    native_language: "Multilingual",
  },
];

function mergeById<T extends { id: string }>(fallback: T[], remote: T[]): T[] {
  const merged = new Map<string, T>();
  fallback.forEach((item) => merged.set(item.id, item));
  remote.forEach((item) => merged.set(item.id, item));
  return Array.from(merged.values());
}

export function LiveEditorPage() {
  const { transcriptionId } = useParams();
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("");
  const [model, setModel] = useState("whisper-base");
  const [task, setTask] = useState<"transcribe" | "translate">("transcribe");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<TranscriptionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ttsModels, setTtsModels] = useState<TTSModel[]>(FALLBACK_TTS_MODELS);
  const [voices, setVoices] = useState<TTSVoice[]>(FALLBACK_TTS_VOICES);
  const [ttsModel, setTtsModel] = useState("Supertone/supertonic-3");
  const [voice, setVoice] = useState("M1");
  const [ttsLanguage, setTtsLanguage] = useState("en");
  const [speed, setSpeed] = useState(1);
  const [synthesizing, setSynthesizing] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [mediaPlaying, setMediaPlaying] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaAudioRef = useRef<HTMLAudioElement>(null);

  const selectedTtsModel = useMemo(
    () => ttsModels.find((item) => item.id === ttsModel),
    [ttsModel, ttsModels],
  );
  const ttsModelFamily = selectedTtsModel?.model_family || "supertonic";
  const supportsVoicePresets =
    selectedTtsModel?.supports_voice_presets !== false;
  const filteredVoices = useMemo(
    () =>
      voices.filter(
        (item) =>
          !item.model_family ||
          item.model_family === "all" ||
          item.model_family === ttsModelFamily,
      ),
    [ttsModelFamily, voices],
  );

  const fileKind = useMemo<"audio" | "video" | null>(() => {
    if (!file) return null;
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (
      file.type.startsWith("video/") ||
      ["mp4", "mov", "mkv", "webm", "avi"].includes(extension || "")
    ) {
      return "video";
    }
    if (
      file.type.startsWith("audio/") ||
      ["mp3", "wav", "flac", "ogg", "m4a", "aac"].includes(extension || "")
    ) {
      return "audio";
    }
    return null;
  }, [file]);

  useEffect(() => {
    let mounted = true;

    async function loadTtsOptions() {
      try {
        const [modelsData, voicesData] = await Promise.all([
          api.getTTSModels(),
          api.getTTSVoices(),
        ]);
        if (!mounted) return;
        setTtsModels(mergeById(FALLBACK_TTS_MODELS, modelsData.models));
        setVoices(mergeById(FALLBACK_TTS_VOICES, voicesData.voices));
        setTtsModel(modelsData.default_model || "Supertone/supertonic-3");
        setVoice(voicesData.default_voice || "M1");
      } catch {
        if (!mounted) return;
        setTtsModels(FALLBACK_TTS_MODELS);
        setVoices(FALLBACK_TTS_VOICES);
      }
    }

    loadTtsOptions();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!transcriptionId) return;

    let mounted = true;
    const existingTranscriptionId = transcriptionId;

    async function loadExistingTranscription() {
      setLoading(true);
      setError(null);
      setFile(null);

      try {
        const item = await api.getTranscription(existingTranscriptionId);
        if (!mounted) return;

        setResult({
          success: true,
          transcription_id: item.id,
          filename: item.filename,
          language: item.result.language,
          text: item.result.text,
          segments: item.result.segments,
          time_taken: item.time_taken,
          model_used: item.model_used,
          model_type: item.model_type,
          is_video: item.is_video,
        });
        setModel(item.model_used);
        setLanguage(item.result.language || "");
      } catch (err) {
        if (!mounted) return;
        setError(
          err instanceof Error ? err.message : "Failed to open transcription",
        );
      } finally {
        if (mounted) setLoading(false);
      }
    }

    loadExistingTranscription();
    return () => {
      mounted = false;
    };
  }, [transcriptionId]);

  useEffect(() => {
    if (!file) {
      setMediaUrl(null);
      setMediaPlaying(false);
      return;
    }

    const nextMediaUrl = URL.createObjectURL(file);
    setMediaUrl(nextMediaUrl);
    setMediaPlaying(false);

    return () => {
      URL.revokeObjectURL(nextMediaUrl);
    };
  }, [file]);

  useEffect(() => {
    if (selectedTtsModel?.languages?.[0]) {
      setTtsLanguage(selectedTtsModel.languages[0]);
    }
  }, [selectedTtsModel]);

  useEffect(() => {
    if (!supportsVoicePresets) {
      setVoice("");
      return;
    }
    if (filteredVoices.length === 0) return;
    if (!filteredVoices.some((item) => item.id === voice)) {
      setVoice(filteredVoices[0].id);
    }
  }, [filteredVoices, supportsVoicePresets, voice]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    const validationError = validateMediaFile(selectedFile);
    if (validationError) {
      setError(validationError);
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setAudioUrl(null);
    setError(null);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError("Please select an audio or video file.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    const progressTimer = window.setInterval(() => {
      setProgress((current) => Math.min(current + Math.random() * 12, 92));
    }, 350);

    try {
      const response = await api.transcribeFile(file, {
        language: language || undefined,
        model,
        task,
      });
      setResult(response);
      setProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcription failed");
    } finally {
      window.clearInterval(progressTimer);
      setLoading(false);
    }
  };

  const handleDownloadTxt = () => {
    if (!result?.text) return;
    downloadBlob(
      new Blob([result.text], { type: "text/plain" }),
      `transcription_${Date.now()}.txt`,
    );
  };

  const handleDownloadSubtitle = async (format: "srt" | "vtt") => {
    if (!result?.transcription_id) return;
    try {
      const { content, filename, mediaType } = await api.downloadSubtitle(
        result.transcription_id,
        format,
      );
      downloadBlob(new Blob([content], { type: mediaType }), filename);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : `Failed to download ${format.toUpperCase()}`,
      );
    }
  };

  const handleSynthesize = async () => {
    if (!result?.text?.trim()) {
      setError("Run a transcription before generating speech.");
      return;
    }

    const currentResult = result;
    const text = currentResult.text.trim();
    setSynthesizing(true);
    setError(null);

    try {
      const { audioBlob } = await api.synthesizeSpeech(text.slice(0, 4000), {
        model: ttsModel,
        voice: supportsVoicePresets ? voice : "",
        speed,
        pitch: 1,
        language: ttsLanguage,
        instruction: null,
        output_format: "wav",
      });
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(audioBlob));
    } catch (err) {
      setError(err instanceof Error ? err.message : "TTS synthesis failed");
    } finally {
      setSynthesizing(false);
    }
  };

  const clearProject = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setAudioUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const toggleMediaPlayback = async () => {
    const audioElement = mediaAudioRef.current;
    if (!audioElement) return;

    if (audioElement.paused) {
      await audioElement.play();
      setMediaPlaying(true);
    } else {
      audioElement.pause();
      setMediaPlaying(false);
    }
  };

  const segments = Array.isArray(result?.segments) ? result.segments : [];
  const transcriptItems = useMemo(() => {
    if (segments.length > 0) {
      return segments.map((segment) => ({
        id: segment.id,
        start: segment.start,
        text: segment.text,
      }));
    }
    if (!result?.text?.trim()) return [];

    return [
      {
        id: "whole-transcript",
        start: null,
        text: result.text.trim(),
      },
    ];
  }, [result?.text, segments]);

  return (
    <div className="studio-grid editor-grid">
      <section className="stage-column">
        <form className="glass-card upload-stage" onSubmit={handleSubmit}>
          <div className="card-header">
            <div>
              <p className="eyebrow">Live Rendering</p>
              <h1>Live Editor</h1>
            </div>
            <span className="badge">
              {result ? "Transcript Ready" : "Input Required"}
            </span>
          </div>

          <div className={`drop-zone ${mediaUrl ? "has-preview" : ""}`}>
            <input
              id="live-editor-file"
              ref={fileInputRef}
              type="file"
              accept={ALLOWED_MEDIA_EXTENSIONS.join(",")}
              onChange={handleFileChange}
              disabled={loading}
            />
            {mediaUrl && fileKind === "video" && (
              <video
                className="drop-media-video"
                src={mediaUrl}
                controls
                preload="metadata"
              />
            )}
            {mediaUrl && fileKind === "audio" && (
              <div className="audio-wave-preview">
                <button
                  type="button"
                  className="preview-play-button"
                  onClick={toggleMediaPlayback}
                >
                  {mediaPlaying ? "Pause" : "Play"}
                </button>
                <div className="inline-waveform" aria-hidden="true">
                  {Array.from({ length: 54 }).map((_, index) => (
                    <span
                      key={index}
                      style={{ height: `${20 + ((index * 19) % 72)}%` }}
                    />
                  ))}
                </div>
                <audio
                  ref={mediaAudioRef}
                  src={mediaUrl}
                  preload="metadata"
                  onPause={() => setMediaPlaying(false)}
                  onPlay={() => setMediaPlaying(true)}
                  onEnded={() => setMediaPlaying(false)}
                />
              </div>
            )}
            <label className="drop-zone-picker" htmlFor="live-editor-file">
              <span>
                {file
                  ? file.name
                  : result?.filename || "Select audio or video media"}
              </span>
              <small>
                {file
                  ? formatFileSize(file.size)
                  : result
                    ? "Opened from Archive. Select a file to run a new transcription."
                    : "MP3, WAV, MP4, MOV, MKV, WEBM up to 50 MB"}
              </small>
            </label>
          </div>

          <div className="control-grid">
            <label>
              Language
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                disabled={loading}
              >
                <option value="">Auto-detect</option>
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="zh">Chinese</option>
                <option value="ja">Japanese</option>
                <option value="ko">Korean</option>
                <option value="pt">Portuguese</option>
                <option value="ru">Russian</option>
              </select>
            </label>
            <label>
              Model
              <select
                value={model}
                onChange={(event) => setModel(event.target.value)}
                disabled={loading}
              >
                <option value="whisper-tiny">Whisper Tiny</option>
                <option value="whisper-base">Whisper Base</option>
                <option value="whisper-small">Whisper Small</option>
                <option value="whisper-medium">Whisper Medium</option>
                <option value="whisper-large">Whisper Large</option>
                <option value="parakeet-tdt-0.6b">Parakeet TDT 0.6B</option>
              </select>
            </label>
            <label>
              Task
              <select
                value={task}
                onChange={(event) =>
                  setTask(event.target.value as "transcribe" | "translate")
                }
                disabled={loading}
              >
                <option value="transcribe">Transcribe</option>
                <option value="translate">Translate to English</option>
              </select>
            </label>
          </div>

          {loading && (
            <div className="progress-block">
              <div className="meter">
                <span style={{ width: `${progress}%` }} />
              </div>
              <small>Processing media... {Math.round(progress)}%</small>
            </div>
          )}

          {error && <div className="alert-card">{error}</div>}

          <div className="button-row">
            <button
              className="studio-button primary"
              type="submit"
              disabled={loading || !file}
            >
              {loading ? "Processing" : "Run Transcription"}
            </button>
            {result && (
              <button
                className="studio-button ghost"
                type="button"
                onClick={clearProject}
              >
                Clear
              </button>
            )}
          </div>
        </form>
      </section>

      <section className="side-column">
        <div className="glass-card transcript-card">
          <div className="card-header">
            <h2>Interactive Transcript</h2>
            <span className="badge">Auto-scroll On</span>
          </div>
          <div className="transcript-scroll">
            {transcriptItems.length === 0 ? (
              <div className="empty-panel">
                {result?.text
                  ? result.text
                  : "Transcript segments will appear here after transcription."}
              </div>
            ) : (
              transcriptItems.map((segment, index) => (
                <article
                  className={`transcript-segment ${index === 1 ? "active" : ""}`}
                  key={segment.id}
                >
                  {typeof segment.start === "number" ? (
                    <time>{formatTime(segment.start)}</time>
                  ) : (
                    <time>Final</time>
                  )}
                  <div>
                    <div className="speaker-row">
                      <span>Speaker_01</span>
                      <small>{index === 0 ? "Introduction" : "Segment"}</small>
                    </div>
                    <p>{segment.text}</p>
                  </div>
                </article>
              ))
            )}
          </div>
          {result && (
            <div className="button-row wrap">
              <button
                className="studio-button ghost"
                type="button"
                onClick={() => navigator.clipboard.writeText(result.text)}
              >
                Copy Text
              </button>
              <button
                className="studio-button ghost"
                type="button"
                onClick={handleDownloadTxt}
              >
                TXT
              </button>
              <button
                className="studio-button ghost"
                type="button"
                onClick={() => handleDownloadSubtitle("srt")}
              >
                SRT
              </button>
              <button
                className="studio-button ghost"
                type="button"
                onClick={() => handleDownloadSubtitle("vtt")}
              >
                VTT
              </button>
            </div>
          )}
        </div>

        <div className="glass-card dubbing-controls">
          <h2>AI Dubbing Controls</h2>
          <div className="control-grid two">
            <label>
              TTS Engine
              <select
                value={ttsModel}
                onChange={(event) => setTtsModel(event.target.value)}
              >
                {ttsModels.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Voice Preset
              {supportsVoicePresets ? (
                <select
                  value={voice}
                  onChange={(event) => setVoice(event.target.value)}
                >
                  {filteredVoices.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name} - {item.native_language || item.language}
                    </option>
                  ))}
                </select>
              ) : (
                <div className="readonly-field">
                  No preset voices. Use voice design attributes.
                </div>
              )}
            </label>
          </div>
          <div className="control-grid two">
            <label>
              Output Language
              <select
                value={ttsLanguage}
                onChange={(event) => setTtsLanguage(event.target.value)}
              >
                {(selectedTtsModel?.languages || ["EN"]).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {selectedTtsModel && (
            <div className="model-feature-panel">
              <strong>{selectedTtsModel.name}</strong>
              <span>{selectedTtsModel.description}</span>
              <ul>
                {(selectedTtsModel.features || [])
                  .slice(0, 4)
                  .map((feature) => (
                    <li key={feature}>{feature}</li>
                  ))}
              </ul>
            </div>
          )}
          <label className="range-control">
            <span>
              Dubbing Speed <strong>{speed.toFixed(2)}x</strong>
            </span>
            <input
              min="0.7"
              max="2"
              step="0.05"
              type="range"
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
            />
          </label>
          <button
            className="studio-button lime"
            type="button"
            onClick={handleSynthesize}
            disabled={synthesizing || !result}
          >
            {synthesizing ? "Synthesizing" : "Run Dubbing Pass"}
          </button>
          {audioUrl && (
            <audio className="audio-player" controls src={audioUrl} />
          )}
        </div>
      </section>
    </div>
  );
}
