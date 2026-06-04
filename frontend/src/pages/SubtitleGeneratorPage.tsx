/**
 * Subtitle Generator Page
 */
import { useRef, useState } from "react";
import * as api from "../api/client";
import { TranscriptionResponse } from "../types";

const ALLOWED_EXTENSIONS = [
  ".mp3",
  ".wav",
  ".flac",
  ".ogg",
  ".m4a",
  ".aac",
  ".mp4",
  ".mov",
  ".mkv",
  ".webm",
  ".avi",
];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export function SubtitleGeneratorPage() {
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("");
  const [model, setModel] = useState("parakeet-tdt-0.6b");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<TranscriptionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [subtitleFormat, setSubtitleFormat] = useState<"srt" | "vtt">("srt");
  const [preview, setPreview] = useState<string | null>(null);
  const [mediaLoading, setMediaLoading] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const ext = "." + selectedFile.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(
        `Invalid file format. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`,
      );
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError("File too large. Maximum size is 50MB.");
      return;
    }

    setFile(selectedFile);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setPreview(null);

    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + Math.random() * 15, 90));
    }, 300);

    try {
      const response = await api.transcribeFile(file, {
        language: language || undefined,
        model: model || undefined,
      });
      clearInterval(progressInterval);
      setProgress(100);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcription failed");
    } finally {
      clearInterval(progressInterval);
      setLoading(false);
    }
  };

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
        err instanceof Error ? err.message : "Failed to download subtitle",
      );
    }
  };

  const handlePreview = async () => {
    if (!result?.transcription_id) return;
    try {
      const { content } = await api.downloadSubtitle(
        result.transcription_id,
        subtitleFormat,
      );
      setPreview(content);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load preview");
    }
  };

  const handleCopyToClipboard = () => {
    if (preview) {
      navigator.clipboard.writeText(preview);
    }
  };

  const handleEmbedVideo = async (mode: "soft" | "hard") => {
    if (!result?.transcription_id) return;
    setMediaLoading(mode);
    setError(null);

    try {
      const { blob, filename } = await api.embedSubtitleVideo(
        result.transcription_id,
        {
          mode,
          format: subtitleFormat,
        },
      );
      downloadBlob(blob, filename);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to generate subtitled video",
      );
    } finally {
      setMediaLoading(null);
    }
  };

  const handleDubVideo = async () => {
    if (!result?.transcription_id) return;
    setMediaLoading("dub");
    setError(null);

    try {
      const { blob, filename } = await api.dubVideo(result.transcription_id, {
        target_language: result.language === "en" ? undefined : "en",
        original_volume: 0.15,
      });
      downloadBlob(blob, filename);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to generate dubbed video",
      );
    } finally {
      setMediaLoading(null);
    }
  };

  const handleReset = () => {
    setFile(null);
    setLanguage("");
    setResult(null);
    setError(null);
    setProgress(0);
    setPreview(null);
    setMediaLoading(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const segmentCount = result?.segments ? result.segments.length : 0;
  const hasVideoOutput = Boolean(result?.is_video && segmentCount > 0);

  return (
    <div className="page">
      <div className="page-header">
        <h1>Subtitle Generator</h1>
        <p>
          Generate SRT/VTT subtitles and video subtitle outputs from timestamped
          speech recognition
        </p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="file">Select Audio or Video File</label>
            <input
              ref={fileInputRef}
              type="file"
              id="file"
              accept={ALLOWED_EXTENSIONS.join(",")}
              onChange={handleFileChange}
              disabled={loading}
            />
            {file && (
              <div className="file-info">
                <span className="name">{file.name}</span>
                <span className="size">{formatFileSize(file.size)}</span>
              </div>
            )}
          </div>

          {loading && (
            <div className="progress-container">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="progress-text">
                Processing... {Math.round(progress)}%
              </p>
            </div>
          )}

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="language">Language</label>
              <select
                id="language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={loading}
              >
                <option value="">Auto-detect</option>
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="it">Italian</option>
                <option value="pt">Portuguese</option>
                <option value="ru">Russian</option>
                <option value="uk">Ukrainian</option>
                <option value="pl">Polish</option>
                <option value="nl">Dutch</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="model">Model</label>
              <select
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={loading}
              >
                <optgroup label="Parakeet TDT Models">
                  <option value="parakeet-tdt-0.6b">
                    Parakeet TDT 0.6B v3 (Recommended)
                  </option>
                </optgroup>
                <optgroup label="Other Models">
                  <option value="whisper-base">Whisper Base</option>
                  <option value="whisper-medium">Whisper Medium</option>
                  <option value="whisper-large">Whisper Large</option>
                </optgroup>
              </select>
              <p className="form-hint">
                Parakeet TDT is optimized for accurate subtitle timestamps.
              </p>
            </div>
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !file}
            >
              {loading ? "Processing..." : "Generate Subtitles"}
            </button>
            {result && (
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

        {result && (
          <div className="result-section">
            <div className="result-header">
              <h3>Subtitles Generated</h3>
              <div className="result-meta">
                <span className="badge">{result.language || "Auto"}</span>
                <span className="badge badge-secondary">
                  {result.model_used}
                </span>
                <span className="badge badge-info">
                  {segmentCount} segments
                </span>
              </div>
            </div>

            <div
              className="result-text"
              style={{ maxHeight: "200px", fontSize: "0.9em" }}
            >
              {result.text.substring(0, 500)}
              {result.text.length > 500 && "..."}
            </div>

            <div className="result-stats">
              <div className="stat">
                <span className="stat-value">{segmentCount}</span>
                <span className="stat-label">Segments</span>
              </div>
              <div className="stat">
                <span className="stat-value">{result.time_taken}s</span>
                <span className="stat-label">Processing Time</span>
              </div>
            </div>

            <div className="result-actions">
              <div className="form-group" style={{ marginBottom: "1rem" }}>
                <label htmlFor="format">Preview Format</label>
                <select
                  id="format"
                  value={subtitleFormat}
                  onChange={(e) =>
                    setSubtitleFormat(e.target.value as "srt" | "vtt")
                  }
                  disabled={loading}
                >
                  <option value="srt">SRT (SubRip - most compatible)</option>
                  <option value="vtt">VTT (WebVTT - web standard)</option>
                </select>
              </div>

              <button className="btn btn-outline" onClick={handlePreview}>
                Preview {subtitleFormat.toUpperCase()}
              </button>
              <button
                className="btn btn-outline"
                onClick={handleCopyToClipboard}
                disabled={!preview}
              >
                Copy to Clipboard
              </button>
              <button
                className="btn btn-outline"
                onClick={() => handleDownloadSubtitle("srt")}
              >
                Download SRT
              </button>
              <button
                className="btn btn-outline"
                onClick={() => handleDownloadSubtitle("vtt")}
              >
                Download VTT
              </button>
              {hasVideoOutput && (
                <>
                  <button
                    className="btn btn-outline"
                    onClick={() => handleEmbedVideo("soft")}
                    disabled={mediaLoading !== null}
                  >
                    {mediaLoading === "soft"
                      ? "Generating..."
                      : "Download Soft-Subtitled Video"}
                  </button>
                  <button
                    className="btn btn-outline"
                    onClick={() => handleEmbedVideo("hard")}
                    disabled={mediaLoading !== null}
                  >
                    {mediaLoading === "hard"
                      ? "Generating..."
                      : "Download Burned-In Video"}
                  </button>
                  <button
                    className="btn btn-outline"
                    onClick={handleDubVideo}
                    disabled={mediaLoading !== null}
                  >
                    {mediaLoading === "dub"
                      ? "Generating..."
                      : "Download Dubbed Video"}
                  </button>
                </>
              )}
            </div>

            {preview && (
              <div className="preview-section">
                <h4>{subtitleFormat.toUpperCase()} Preview</h4>
                <pre className="subtitle-preview">{preview}</pre>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="info-cards">
        <div className="info-card">
          <h4>Precise Timestamps</h4>
          <p>
            Parakeet TDT uses transducer architecture for accurate subtitle
            timing
          </p>
        </div>
        <div className="info-card">
          <h4>24+ Languages</h4>
          <p>English, European languages, Russian, Ukrainian supported</p>
        </div>
        <div className="info-card">
          <h4>Video Outputs</h4>
          <p>
            Create SRT/VTT files, embedded subtitle videos, and first-pass
            dubbed videos
          </p>
        </div>
      </div>
    </div>
  );
}
