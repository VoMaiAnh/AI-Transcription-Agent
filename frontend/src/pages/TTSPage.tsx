/**
 * Text-to-Speech Page
 */
import { useMemo, useState, useEffect } from "react";
import * as api from "../api/client";
import { TTSVoice } from "../types";

const SUPERTONIC_MODEL = { id: "Supertone/supertonic-3", name: "Supertonic 3" };

const SUPERTONIC_LANGUAGES = [
  { code: "en", label: "English" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "ja", label: "Japanese" },
  { code: "ko", label: "Korean" },
  { code: "pt", label: "Portuguese" },
  { code: "ru", label: "Russian" },
  { code: "ar", label: "Arabic" },
  { code: "bg", label: "Bulgarian" },
  { code: "cs", label: "Czech" },
  { code: "da", label: "Danish" },
  { code: "el", label: "Greek" },
  { code: "et", label: "Estonian" },
  { code: "fi", label: "Finnish" },
  { code: "hi", label: "Hindi" },
  { code: "hr", label: "Croatian" },
  { code: "hu", label: "Hungarian" },
  { code: "id", label: "Indonesian" },
  { code: "it", label: "Italian" },
  { code: "lt", label: "Lithuanian" },
  { code: "lv", label: "Latvian" },
  { code: "nl", label: "Dutch" },
  { code: "pl", label: "Polish" },
  { code: "ro", label: "Romanian" },
  { code: "sk", label: "Slovak" },
  { code: "sl", label: "Slovenian" },
  { code: "sv", label: "Swedish" },
  { code: "tr", label: "Turkish" },
  { code: "uk", label: "Ukrainian" },
  { code: "vi", label: "Vietnamese" },
  { code: "na", label: "Unknown / fallback" },
];

const FALLBACK_VOICES: TTSVoice[] = [
  {
    id: "M1",
    name: "Supertonic M1",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "M2",
    name: "Supertonic M2",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "M3",
    name: "Supertonic M3",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "M4",
    name: "Supertonic M4",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "M5",
    name: "Supertonic M5",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "F1",
    name: "Supertonic F1",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "F2",
    name: "Supertonic F2",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "F3",
    name: "Supertonic F3",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "F4",
    name: "Supertonic F4",
    language: "multilingual",
    model_family: "supertonic",
  },
  {
    id: "F5",
    name: "Supertonic F5",
    language: "multilingual",
    model_family: "supertonic",
  },
];

export function TTSPage() {
  const [text, setText] = useState("");
  const [model, setModel] = useState(SUPERTONIC_MODEL.id);
  const [voice, setVoice] = useState("M1");
  const [speed, setSpeed] = useState(1.0);
  const [language, setLanguage] = useState("en");
  const [outputFormat, setOutputFormat] = useState<"wav" | "mp3">("wav");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);

  const [availableVoices, setAvailableVoices] =
    useState<TTSVoice[]>(FALLBACK_VOICES);

  useEffect(() => {
    api
      .getTTSVoices()
      .then((res) => setAvailableVoices(res.voices))
      .catch(() => setAvailableVoices(FALLBACK_VOICES));
  }, []);

  const filteredVoices = useMemo(() => availableVoices, [availableVoices]);

  useEffect(() => {
    if (!filteredVoices.some((item) => item.id === voice)) {
      setVoice(filteredVoices[0]?.id || "M1");
    }
  }, [filteredVoices, voice]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!text.trim()) {
      setError("Please enter text to synthesize");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const { audioBlob, duration: audioDuration } = await api.synthesizeSpeech(
        text,
        {
          model,
          voice,
          speed,
          pitch: 1,
          language,
          instruction: null,
          output_format: outputFormat,
        },
      );

      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      setDuration(audioDuration);
    } catch (err) {
      setError(err instanceof Error ? err.message : "TTS synthesis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setText("");
    setAudioUrl(null);
    setDuration(null);
    setError(null);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Text-to-Speech</h1>
        <p>Convert text to natural-sounding speech using AI voice synthesis</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="text">Text to Synthesize</label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the text you want to convert to speech..."
              maxLength={5000}
              disabled={loading}
              rows={6}
            />
            <div className="char-count">{text.length} / 5000 characters</div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="model">TTS Model</label>
              <select
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={loading}
              >
                <optgroup label="Supertonic Models">
                  <option value={SUPERTONIC_MODEL.id}>
                    {SUPERTONIC_MODEL.name}
                  </option>
                </optgroup>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="voice">Voice</label>
              <select
                id="voice"
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
                disabled={loading}
              >
                {filteredVoices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} ({v.language})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="language">Language</label>
              <select
                id="language"
                value={language}
                onChange={(e) => {
                  setLanguage(e.target.value);
                }}
                disabled={loading}
              >
                {SUPERTONIC_LANGUAGES.map((item) => (
                  <option key={item.code} value={item.code}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Speed: {speed.toFixed(1)}x</label>
              <input
                type="range"
                min="0.7"
                max="2.0"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                disabled={loading}
              />
              <div className="range-labels">
                <span>0.7x</span>
                <span>2.0x</span>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="format">Format</label>
              <select
                id="format"
                value={outputFormat}
                onChange={(e) =>
                  setOutputFormat(e.target.value as "wav" | "mp3")
                }
                disabled={loading}
              >
                <option value="wav">WAV (Uncompressed)</option>
                <option value="mp3">MP3 (Compressed)</option>
              </select>
            </div>
          </div>

          {error && (
            <div className="alert alert-error">
              <span>⚠️</span> {error}
            </div>
          )}

          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !text.trim()}
            >
              {loading ? "Synthesizing..." : "Synthesize"}
            </button>
            {audioUrl && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleReset}
              >
                Clear
              </button>
            )}
          </div>
        </form>

        {audioUrl && (
          <div className="result-section">
            <div className="result-header">
              <h3>Generated Audio</h3>
              {duration && (
                <span className="badge">Duration: {duration.toFixed(2)}s</span>
              )}
            </div>

            <div className="audio-container">
              <audio
                controls
                src={audioUrl}
                autoPlay
                className="audio-player"
              />
            </div>

            <div className="result-actions">
              <a
                href={audioUrl}
                download={`tts_${Date.now()}.${outputFormat}`}
                className="btn btn-outline"
              >
                💾 Download Audio
              </a>
            </div>
          </div>
        )}
      </div>

      <div className="info-cards">
        <div className="info-card">
          <h4>Voice Options</h4>
          <p>Built-in Supertonic voice styles</p>
        </div>
        <div className="info-card">
          <h4>Speed Control</h4>
          <p>Adjust playback speed (0.7x - 2.0x)</p>
        </div>
      </div>
    </div>
  );
}
