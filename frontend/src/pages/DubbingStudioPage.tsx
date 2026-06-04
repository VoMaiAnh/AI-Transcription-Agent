import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import * as api from "../api/client";
import { TranscriptionInfo } from "../types";
import { downloadBlob, formatDate, formatTime } from "./pageUtils";

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
  { code: "hi", label: "Hindi" },
  { code: "it", label: "Italian" },
  { code: "nl", label: "Dutch" },
  { code: "pl", label: "Polish" },
  { code: "tr", label: "Turkish" },
  { code: "uk", label: "Ukrainian" },
  { code: "vi", label: "Vietnamese" },
  { code: "na", label: "Unknown / fallback" },
];

const SUPERTONIC_VOICES = [
  { id: "M1", name: "Supertonic M1" },
  { id: "M2", name: "Supertonic M2" },
  { id: "M3", name: "Supertonic M3" },
  { id: "M4", name: "Supertonic M4" },
  { id: "M5", name: "Supertonic M5" },
  { id: "F1", name: "Supertonic F1" },
  { id: "F2", name: "Supertonic F2" },
  { id: "F3", name: "Supertonic F3" },
  { id: "F4", name: "Supertonic F4" },
  { id: "F5", name: "Supertonic F5" },
];

type RenderAction = "soft" | "hard" | "dub" | "combined-soft" | "combined-hard";

function chooseRenderedMediaKey(
  mediaPaths?: Record<string, string>,
): string | null {
  const keys = Object.keys(mediaPaths || {});
  if (keys.length === 0) return null;

  return (
    keys.find((key) => key.startsWith("dubbed_")) ||
    keys.find(
      (key) =>
        key.includes("subtitles_hard") || key.includes("subtitles_burned"),
    ) ||
    keys.find((key) => key.includes("subtitles_soft")) ||
    keys.find((key) => key.includes("subtitles")) ||
    keys[0]
  );
}

export function DubbingStudioPage() {
  const { transcriptionId } = useParams();
  const [projects, setProjects] = useState<TranscriptionInfo[]>([]);
  const [project, setProject] = useState<TranscriptionInfo | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [renderedUrl, setRenderedUrl] = useState<string | null>(null);
  const [renderedLabel, setRenderedLabel] = useState("Rendered output");
  const [renderedDownload, setRenderedDownload] = useState<{
    blob: Blob;
    filename: string;
  } | null>(null);
  const [lastRenderAction, setLastRenderAction] = useState<RenderAction | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [mediaLoading, setMediaLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [subtitleFormat, setSubtitleFormat] = useState<"srt" | "vtt">("srt");
  const [preview, setPreview] = useState<string | null>(null);
  const [trackLanguage, setTrackLanguage] = useState("original");
  const [voice, setVoice] = useState("M1");
  const [ttsModel, setTtsModel] = useState(SUPERTONIC_MODEL.id);
  const [speed, setSpeed] = useState(1);
  const [originalVolume, setOriginalVolume] = useState(0.15);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [enableSubtitles, setEnableSubtitles] = useState(true);
  const [enableDubbing, setEnableDubbing] = useState(true);
  const [replaceOriginalAudio, setReplaceOriginalAudio] = useState(false);
  const [generatingTrackAudio, setGeneratingTrackAudio] = useState(false);
  const blobUrlRef = useRef<string | null>(null);

  const clearBlobPreview = () => {
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
  };

  const loadProject = async (id: string) => {
    setLoading(true);
    setError(null);
    setPreview(null);
    setRenderedDownload(null);
    setLastRenderAction(null);
    clearBlobPreview();

    try {
      const item = await api.getTranscription(id);
      setProject(item);
      setSourceUrl(api.getTranscriptionSourceUrl(item.id));
      setTrackLanguage("original");

      const mediaKey = chooseRenderedMediaKey(item.media_paths);
      if (mediaKey) {
        setRenderedUrl(api.getTranscriptionMediaUrl(item.id, mediaKey));
        setRenderedLabel(mediaKey.replace(/_/g, " "));
      } else {
        setRenderedUrl(null);
        setRenderedLabel("Rendered output");
      }
    } catch (err) {
      setProject(null);
      setSourceUrl(null);
      setRenderedUrl(null);
      setError(err instanceof Error ? err.message : "Failed to load project");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let mounted = true;

    async function loadProjects() {
      setLoading(true);
      setError(null);

      try {
        const data = await api.listTranscriptions();
        if (!mounted) return;
        const sorted = [...data.transcriptions].sort(
          (a, b) =>
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
        );
        setProjects(sorted);

        const initialId =
          transcriptionId ||
          sorted.find((item) => item.is_video)?.id ||
          sorted[0]?.id;
        if (initialId) {
          await loadProject(initialId);
        } else {
          setProject(null);
          setSourceUrl(null);
          setRenderedUrl(null);
          setLoading(false);
        }
      } catch (err) {
        if (!mounted) return;
        setError(
          err instanceof Error ? err.message : "Failed to load earlier work",
        );
        setLoading(false);
      }
    }

    loadProjects();
    return () => {
      mounted = false;
      clearBlobPreview();
    };
  }, [transcriptionId]);

  const translations = project?.translations || {};
  const translationEntries = Object.values(translations);
  const selectedTranslation =
    trackLanguage === "original" ? null : translations[trackLanguage] || null;
  const segments =
    selectedTranslation?.segments || project?.result.segments || [];
  const canExportVideo = Boolean(project?.is_video && segments.length > 0);
  const selectedTrackHasAudio =
    trackLanguage === "original" ||
    Boolean(selectedTranslation?.tts_audio_path);
  const renderedMediaKeys = Object.keys(project?.media_paths || {});
  const effectiveOriginalVolume = replaceOriginalAudio ? 0 : originalVolume;
  const trackLabel =
    trackLanguage === "original"
      ? "Original"
      : SUPERTONIC_LANGUAGES.find((item) => item.code === trackLanguage)
          ?.label || trackLanguage.toUpperCase();

  const handlePreview = async () => {
    if (!project) return;
    try {
      const { content } = await api.downloadSubtitle(
        project.id,
        subtitleFormat,
        trackLanguage,
      );
      setPreview(content);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load subtitle preview",
      );
    }
  };

  const handleDownloadSubtitle = async () => {
    if (!project) return;
    try {
      const { content, filename, mediaType } = await api.downloadSubtitle(
        project.id,
        subtitleFormat,
        trackLanguage,
      );
      downloadBlob(new Blob([content], { type: mediaType }), filename);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to download subtitle track",
      );
    }
  };

  const setRenderedBlob = (blob: Blob, filename: string, label: string) => {
    clearBlobPreview();
    const nextUrl = URL.createObjectURL(blob);
    blobUrlRef.current = nextUrl;
    setRenderedUrl(nextUrl);
    setRenderedLabel(label);
    setRenderedDownload({ blob, filename });
  };

  const handleEmbedVideo = async (mode: "soft" | "hard") => {
    if (!project) return;
    setMediaLoading(mode);
    setError(null);

    try {
      const { blob, filename } = await api.embedSubtitleVideo(project.id, {
        mode,
        format: subtitleFormat,
        language: trackLanguage,
      });
      setRenderedBlob(
        blob,
        filename,
        mode === "hard"
          ? "Hard-burned subtitle render"
          : "Soft subtitle render",
      );
      setLastRenderAction(mode);
      const refreshed = await api.getTranscription(project.id);
      setProject(refreshed);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to render subtitled video",
      );
    } finally {
      setMediaLoading(null);
    }
  };

  const handleDubVideo = async () => {
    if (!project) return;
    setMediaLoading("dub");
    setError(null);

    try {
      const { blob, filename } = await api.dubVideo(project.id, {
        language: trackLanguage,
        tts_model: ttsModel,
        voice,
        speed,
        pitch: 1,
        original_volume: effectiveOriginalVolume,
        whisper_model: "whisper-base",
      });
      setRenderedBlob(blob, filename, "Final dubbed render");
      setLastRenderAction("dub");
      const refreshed = await api.getTranscription(project.id);
      setProject(refreshed);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to render final dub",
      );
    } finally {
      setMediaLoading(null);
    }
  };

  const handleCombinedVideo = async (mode: "soft" | "hard") => {
    if (!project) return;
    const loadingKey = mode === "hard" ? "combined-hard" : "combined-soft";
    setMediaLoading(loadingKey);
    setError(null);

    try {
      const { blob, filename } = await api.dubAndSubtitleVideo(project.id, {
        language: trackLanguage,
        subtitle_mode: mode,
        subtitle_format: subtitleFormat,
        tts_model: ttsModel,
        voice,
        speed,
        pitch: 1,
        original_volume: effectiveOriginalVolume,
        whisper_model: "whisper-base",
      });
      setRenderedBlob(
        blob,
        filename,
        mode === "hard"
          ? "Dubbed hard-burned subtitle render"
          : "Dubbed soft subtitle render",
      );
      setLastRenderAction(mode === "hard" ? "combined-hard" : "combined-soft");
      const refreshed = await api.getTranscription(project.id);
      setProject(refreshed);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to render dubbed subtitled video",
      );
    } finally {
      setMediaLoading(null);
    }
  };

  const handleGenerateTrackAudio = async (replaceExisting = false) => {
    if (!project || trackLanguage === "original") return;
    setGeneratingTrackAudio(true);
    setError(null);

    try {
      const response = await api.generateTranslatedTTS(project.id, {
        target_language: trackLanguage,
        tts_model: ttsModel,
        voice,
        speed,
        replace_existing: replaceExisting,
      });
      const refreshed = await api.getTranscription(project.id);
      setProject({
        ...refreshed,
        translations: {
          ...(refreshed.translations || {}),
          [response.translation.language]: response.translation,
        },
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to generate translated audio",
      );
    } finally {
      setGeneratingTrackAudio(false);
    }
  };

  const handleRenderedArtifactSelect = (mediaKey: string) => {
    if (!project) return;
    clearBlobPreview();
    setRenderedDownload(null);
    setLastRenderAction(null);
    setRenderedUrl(api.getTranscriptionMediaUrl(project.id, mediaKey));
    setRenderedLabel(mediaKey.replace(/_/g, " "));
  };

  const handleRenderedDownload = async () => {
    if (renderedDownload) {
      downloadBlob(renderedDownload.blob, renderedDownload.filename);
      return;
    }

    if (!renderedUrl || !project) return;
    try {
      const response = await fetch(renderedUrl);
      if (!response.ok) throw new Error("Failed to download rendered media");
      const blob = await response.blob();
      const extension = blob.type.includes("matroska") ? "mkv" : "mp4";
      const safeLabel =
        renderedLabel
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-|-$/g, "") || "render";
      downloadBlob(blob, `${project.id}_${safeLabel}.${extension}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to download final result",
      );
    }
  };

  const handleRedoRender = async () => {
    if (!lastRenderAction) return;
    if (lastRenderAction === "dub") {
      await handleDubVideo();
    } else if (lastRenderAction === "combined-soft") {
      await handleCombinedVideo("soft");
    } else if (lastRenderAction === "combined-hard") {
      await handleCombinedVideo("hard");
    } else {
      await handleEmbedVideo(lastRenderAction);
    }
  };

  return (
    <div
      className={`studio-grid dubbing-grid ${sidebarOpen ? "" : "sidebar-collapsed"}`}
    >
      <button
        className="studio-button ghost sidebar-toggle"
        type="button"
        onClick={() => setSidebarOpen((current) => !current)}
      >
        {sidebarOpen ? "Hide Library" : "Show Library"}
      </button>

      <aside
        className={`glass-card voice-library ${sidebarOpen ? "" : "hidden"}`}
      >
        <h2>Project Library</h2>
        <div className="model-note">
          <strong>{projects.length} saved jobs</strong>
          <span>Open transcription work created in Live Editor.</span>
        </div>
        {projects.length === 0 ? (
          <div className="empty-panel small">
            <span>No saved transcriptions yet.</span>
            <Link className="archive-link-button" to="/editor">
              Open Live Editor
            </Link>
          </div>
        ) : (
          <div className="project-library-list">
            {projects.map((item) => (
              <button
                key={item.id}
                className={`library-project-button ${project?.id === item.id ? "active" : ""}`}
                type="button"
                onClick={() => loadProject(item.id)}
              >
                <span className="media-thumb small">
                  {item.is_video ? "AV" : "AU"}
                </span>
                <span>
                  <strong>{item.filename}</strong>
                  <small>
                    {item.result.language || "Auto"} -{" "}
                    {formatDate(item.created_at)}
                  </small>
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="model-note">
          <strong>Supertonic 3</strong>
          <span>Built-in voice styles for fast on-device dubbing.</span>
        </div>
        <div className="voice-list-compact">
          {SUPERTONIC_VOICES.map((item) => (
            <button
              key={item.id}
              className={`voice-card ${voice === item.id ? "active" : ""}`}
              type="button"
              onClick={() => setVoice(item.id)}
            >
              <span className="avatar-token">{item.id}</span>
              <span>{item.name}</span>
            </button>
          ))}
        </div>
      </aside>

      <section className="stage-column">
        <section className="glass-card media-workbench">
          <div className="card-header">
            <div>
              <p className="eyebrow">Timeline Master</p>
              <h1>Dubbing & Subtitle Studio</h1>
            </div>
            <span className="badge">
              {project ? "Project Loaded" : loading ? "Loading" : "No Project"}
            </span>
          </div>

          {error && <div className="alert-card">{error}</div>}

          {project ? (
            <>
              <div className="workflow-mode-row">
                <label
                  className={`mode-toggle ${enableSubtitles ? "active" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={enableSubtitles}
                    onChange={(event) =>
                      setEnableSubtitles(event.target.checked)
                    }
                  />
                  <span>Add subtitles</span>
                </label>
                <label
                  className={`mode-toggle ${enableDubbing ? "active" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={enableDubbing}
                    onChange={(event) => setEnableDubbing(event.target.checked)}
                  />
                  <span>Add dubbings</span>
                </label>
              </div>

              <div className="studio-preview-grid">
                <div className="media-preview-frame">
                  <div className="preview-label">
                    <strong>Source Media</strong>
                    <span>{project.filename}</span>
                  </div>
                  {sourceUrl && project.is_video ? (
                    <video
                      className="studio-video-preview"
                      src={sourceUrl}
                      controls
                      preload="metadata"
                    />
                  ) : sourceUrl ? (
                    <audio
                      className="audio-player"
                      src={sourceUrl}
                      controls
                      preload="metadata"
                    />
                  ) : (
                    <div className="empty-panel small">
                      Source media is unavailable.
                    </div>
                  )}
                </div>

                <div className="media-preview-frame">
                  <div className="preview-label">
                    <strong>Rendered Video</strong>
                    <span>
                      {renderedUrl
                        ? renderedLabel
                        : "Generate subtitles or dubbing to preview output"}
                    </span>
                  </div>
                  {renderedUrl ? (
                    <>
                      <video
                        className="studio-video-preview"
                        src={renderedUrl}
                        controls
                        preload="metadata"
                      />
                      <div className="button-row wrap">
                        <button
                          className="studio-button ghost small"
                          type="button"
                          onClick={handleRenderedDownload}
                        >
                          Download Final Result
                        </button>
                        <button
                          className="studio-button ghost small"
                          type="button"
                          onClick={handleRedoRender}
                          disabled={!lastRenderAction || mediaLoading !== null}
                        >
                          Redo Render
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="empty-panel small">
                      No rendered media yet.
                    </div>
                  )}
                </div>
              </div>

              <div className="project-meta-strip">
                <span>
                  {project.is_video ? "Video source" : "Audio source"}
                </span>
                <span>{segments.length} timed segments</span>
                <span>{trackLabel} track</span>
                <span>{translationEntries.length} translations</span>
                <span>{project.model_used}</span>
                <span>{project.result.language || "Auto language"}</span>
              </div>
            </>
          ) : (
            <div className="empty-panel">
              <strong>No Live Editor work loaded</strong>
              <span>
                Run a transcription in Live Editor, then open it here for
                subtitle and dubbing delivery.
              </span>
              <Link className="studio-button primary" to="/editor">
                Open Live Editor
              </Link>
            </div>
          )}
        </section>

        <section className="glass-card timeline-card">
          <div className="card-header">
            <h2>Timeline Master</h2>
            <span className="badge">Auto-Sync</span>
          </div>
          <div className="time-ruler">
            <span>00:00</span>
            <span>00:10</span>
            <span>00:20</span>
            <span>00:30</span>
            <span>00:40</span>
          </div>
          <div className="timeline-track video-track">
            <span>{project?.filename || "Source media track"}</span>
          </div>
          <div className="timeline-track audio-track">
            {Array.from({ length: 48 }).map((_, index) => (
              <i
                key={index}
                style={{ height: `${15 + ((index * 13) % 65)}%` }}
              />
            ))}
          </div>
          <div className="timeline-track dub-track">
            <span>
              {segments.length ? `Dubbing: ${voice}` : "Dubbed audio track"}
            </span>
          </div>
          <div className="segment-strip">
            {segments.slice(0, 8).map((segment) => (
              <span key={segment.id}>{formatTime(segment.start)}</span>
            ))}
          </div>
        </section>

        {preview && (
          <section className="glass-card preview-card">
            <div className="card-header">
              <h2>{subtitleFormat.toUpperCase()} Preview</h2>
              <button
                className="studio-button ghost small"
                type="button"
                onClick={() => navigator.clipboard.writeText(preview)}
              >
                Copy
              </button>
            </div>
            <pre>{preview}</pre>
          </section>
        )}
      </section>

      <aside className="side-column">
        <section className="glass-card config-card">
          <h2>Dubbing Config</h2>
          <label>
            Track Language
            <select
              value={trackLanguage}
              onChange={(event) => setTrackLanguage(event.target.value)}
            >
              <option value="original">Original transcript</option>
              {translationEntries.map((translation) => (
                <option key={translation.language} value={translation.language}>
                  {SUPERTONIC_LANGUAGES.find(
                    (item) => item.code === translation.language,
                  )?.label || translation.language.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          {trackLanguage !== "original" && (
            <div className="translation-status">
              <strong>{trackLabel} translation</strong>
              <span>
                {selectedTranslation
                  ? `${selectedTranslation.segments.length} segments - ${
                      selectedTranslation.tts_audio_path
                        ? "audio ready"
                        : "audio required before final dub"
                    }`
                  : "Select a saved Live Studio translation."}
              </span>
            </div>
          )}
          <label>
            TTS Model
            <select
              value={ttsModel}
              onChange={(event) => setTtsModel(event.target.value)}
            >
              <option value={SUPERTONIC_MODEL.id}>
                {SUPERTONIC_MODEL.name}
              </option>
            </select>
          </label>
          <label>
            Voice
            <select
              value={voice}
              onChange={(event) => setVoice(event.target.value)}
            >
              {SUPERTONIC_VOICES.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label className="range-control">
            <span>
              Speed <strong>{speed.toFixed(2)}x</strong>
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
          <label className="range-control">
            <span>
              Original Volume{" "}
              <strong>{Math.round(originalVolume * 100)}%</strong>
            </span>
            <input
              min="0"
              max="1"
              step="0.05"
              type="range"
              value={originalVolume}
              onChange={(event) =>
                setOriginalVolume(Number(event.target.value))
              }
              disabled={replaceOriginalAudio}
            />
          </label>
          <label
            className={`mode-toggle audio-replace-toggle ${replaceOriginalAudio ? "active" : ""}`}
          >
            <input
              type="checkbox"
              checked={replaceOriginalAudio}
              onChange={(event) =>
                setReplaceOriginalAudio(event.target.checked)
              }
            />
            <span>Replace original audio with dub</span>
          </label>
        </section>

        <section className="glass-card export-card">
          <h2>Export & Deliver</h2>
          {enableSubtitles && (
            <label>
              Subtitle Format
              <select
                value={subtitleFormat}
                onChange={(event) =>
                  setSubtitleFormat(event.target.value as "srt" | "vtt")
                }
              >
                <option value="srt">SRT</option>
                <option value="vtt">VTT</option>
              </select>
            </label>
          )}
          {renderedMediaKeys.length > 0 && (
            <label>
              Existing Renders
              <select
                value=""
                onChange={(event) =>
                  event.target.value &&
                  handleRenderedArtifactSelect(event.target.value)
                }
              >
                <option value="">Open saved render</option>
                {renderedMediaKeys.map((key) => (
                  <option key={key} value={key}>
                    {key.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </label>
          )}
          {renderedUrl && (
            <div className="button-row wrap">
              <button
                className="studio-button ghost"
                type="button"
                onClick={handleRenderedDownload}
              >
                Download Final Result
              </button>
              <button
                className="studio-button ghost"
                type="button"
                onClick={handleRedoRender}
                disabled={!lastRenderAction || mediaLoading !== null}
              >
                Redo Render
              </button>
            </div>
          )}
          {enableSubtitles && (
            <>
              <button
                className="export-option"
                type="button"
                disabled={!project}
                onClick={handlePreview}
              >
                <strong>Preview Subtitle Tracks</strong>
                <span>SRT/VTT transcript timing</span>
              </button>
              <button
                className="export-option"
                type="button"
                disabled={!project}
                onClick={handleDownloadSubtitle}
              >
                <strong>Download Subtitle Tracks</strong>
                <span>Deliver standalone captions</span>
              </button>
              {!enableDubbing && (
                <>
                  <button
                    className="export-option"
                    type="button"
                    disabled={!canExportVideo || mediaLoading !== null}
                    onClick={() => handleEmbedVideo("soft")}
                  >
                    <strong>
                      {mediaLoading === "soft"
                        ? "Rendering..."
                        : "Preview Soft Subtitles"}
                    </strong>
                    <span>Generate selectable subtitle video</span>
                  </button>
                  <button
                    className="export-option"
                    type="button"
                    disabled={!canExportVideo || mediaLoading !== null}
                    onClick={() => handleEmbedVideo("hard")}
                  >
                    <strong>
                      {mediaLoading === "hard"
                        ? "Rendering..."
                        : "Preview Hard Burn"}
                    </strong>
                    <span>Generate permanent subtitle render</span>
                  </button>
                </>
              )}
            </>
          )}
          {enableSubtitles && enableDubbing && (
            <>
              <button
                className="export-option"
                type="button"
                disabled={
                  !canExportVideo ||
                  mediaLoading !== null ||
                  generatingTrackAudio ||
                  !selectedTrackHasAudio
                }
                onClick={() => handleCombinedVideo("soft")}
              >
                <strong>
                  {mediaLoading === "combined-soft"
                    ? "Rendering dub + soft subtitles..."
                    : "Render Dub + Soft Subtitles"}
                </strong>
                <span>
                  {trackLabel} audio and selectable subtitles in one render
                </span>
              </button>
              <button
                className="export-option"
                type="button"
                disabled={
                  !canExportVideo ||
                  mediaLoading !== null ||
                  generatingTrackAudio ||
                  !selectedTrackHasAudio
                }
                onClick={() => handleCombinedVideo("hard")}
              >
                <strong>
                  {mediaLoading === "combined-hard"
                    ? "Rendering dub + hard burn..."
                    : "Render Dub + Hard Burn"}
                </strong>
                <span>
                  {trackLabel} audio and permanent subtitles in one render
                </span>
              </button>
            </>
          )}
          {enableDubbing && (
            <>
              {trackLanguage !== "original" && !selectedTrackHasAudio && (
                <button
                  className="export-option"
                  type="button"
                  disabled={!selectedTranslation || generatingTrackAudio}
                  onClick={() => handleGenerateTrackAudio(false)}
                >
                  <strong>
                    {generatingTrackAudio
                      ? "Generating translated audio..."
                      : "Generate Translated Audio"}
                  </strong>
                  <span>Required before rendering the translated dub</span>
                </button>
              )}
              {trackLanguage !== "original" && selectedTrackHasAudio && (
                <button
                  className="export-option"
                  type="button"
                  disabled={generatingTrackAudio}
                  onClick={() => handleGenerateTrackAudio(true)}
                >
                  <strong>
                    {generatingTrackAudio
                      ? "Regenerating translated audio..."
                      : "Redo Translated Audio"}
                  </strong>
                  <span>
                    Refresh the saved dub track with current voice settings
                  </span>
                </button>
              )}
              {!enableSubtitles && (
                <button
                  className="studio-button primary full"
                  type="button"
                  disabled={
                    !canExportVideo ||
                    mediaLoading !== null ||
                    generatingTrackAudio ||
                    !selectedTrackHasAudio
                  }
                  onClick={handleDubVideo}
                >
                  {mediaLoading === "dub"
                    ? "Rendering Final Dub"
                    : replaceOriginalAudio
                      ? "Render Dub and Replace Audio"
                      : "Render Final Dub"}
                </button>
              )}
            </>
          )}
          {!enableSubtitles && !enableDubbing && (
            <div className="empty-panel small">
              Choose subtitles, dubbing, or both to render an output.
            </div>
          )}
        </section>
      </aside>
    </div>
  );
}
