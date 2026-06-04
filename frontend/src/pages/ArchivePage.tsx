import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import * as api from "../api/client";
import { TranscriptionInfo, TTSCacheEntry } from "../types";
import { downloadBlob, formatDate } from "./pageUtils";

type ArchiveType = "all" | "transcription" | "tts";

type ArchiveItem = {
  id: string;
  type: "transcription" | "tts";
  name: string;
  model: string;
  createdAt: string;
  detail: string;
  hasSubtitles: boolean;
};

export function ArchivePage() {
  const [transcriptions, setTranscriptions] = useState<TranscriptionInfo[]>([]);
  const [ttsResults, setTtsResults] = useState<TTSCacheEntry[]>([]);
  const [filter, setFilter] = useState<ArchiveType>("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadArchive = async () => {
    setLoading(true);
    setError(null);
    try {
      const [transcriptionData, ttsData] = await Promise.all([
        api.listTranscriptions(),
        api.listTTSResults(),
      ]);
      setTranscriptions(transcriptionData.transcriptions);
      setTtsResults(ttsData.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load archive");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadArchive();
  }, []);

  const items = useMemo<ArchiveItem[]>(() => {
    const transcriptionItems = transcriptions.map((item) => ({
      id: item.id,
      type: "transcription" as const,
      name: item.filename,
      model: item.model_used,
      createdAt: item.created_at,
      detail: `${item.is_video ? "Video" : "Audio"} · ${item.result.language || "Auto"} · ${item.time_taken}s`,
      hasSubtitles: item.result.segments.length > 0,
    }));

    const ttsItems = ttsResults.map((item) => ({
      id: item.id,
      type: "tts" as const,
      name: item.text.slice(0, 80) || `TTS result ${item.id}`,
      model: item.model,
      createdAt: item.created_at,
      detail: `${item.voice} · ${item.duration.toFixed(2)}s · ${item.language || "Auto"}`,
      hasSubtitles: false,
    }));

    return [...transcriptionItems, ...ttsItems]
      .filter((item) => filter === "all" || item.type === filter)
      .filter((item) => {
        const haystack =
          `${item.name} ${item.model} ${item.detail}`.toLowerCase();
        return haystack.includes(query.toLowerCase());
      })
      .sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      );
  }, [filter, query, transcriptions, ttsResults]);

  const handleDelete = async (item: ArchiveItem) => {
    if (!window.confirm(`Delete ${item.name}?`)) return;
    try {
      if (item.type === "transcription") {
        await api.deleteTranscription(item.id);
      } else {
        await api.deleteTTSResult(item.id);
      }
      await loadArchive();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to delete archive item",
      );
    }
  };

  const handleSubtitleDownload = async (
    item: ArchiveItem,
    format: "srt" | "vtt",
  ) => {
    try {
      const { content, filename, mediaType } = await api.downloadSubtitle(
        item.id,
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

  return (
    <div className="archive-page">
      <section className="page-heading archive-heading">
        <div>
          <p className="eyebrow">Deep Search Index</p>
          <h1>Archive & History</h1>
          <p>
            Access and manage historical transcription and text-to-speech
            projects.
          </p>
        </div>
        <div className="glass-card total-card">
          <span>Total Projects</span>
          <strong>{transcriptions.length + ttsResults.length}</strong>
        </div>
      </section>

      {error && <div className="alert-card">{error}</div>}

      <section className="glass-card archive-filters">
        <div className="segmented-control">
          {(["all", "transcription", "tts"] as ArchiveType[]).map((item) => (
            <button
              key={item}
              className={filter === item ? "active" : ""}
              type="button"
              onClick={() => setFilter(item)}
            >
              {item === "all"
                ? "All Files"
                : item === "tts"
                  ? "TTS"
                  : "Transcriptions"}
            </button>
          ))}
        </div>
        <label className="archive-search">
          Search Archive
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filename, model, text..."
          />
        </label>
        <button
          className="studio-button ghost"
          type="button"
          onClick={loadArchive}
        >
          Refresh
        </button>
      </section>

      <section className="glass-card archive-table-card">
        {loading ? (
          <div className="loading-panel">Loading archive...</div>
        ) : items.length === 0 ? (
          <div className="empty-panel">
            No archive items match the current filters.
          </div>
        ) : (
          <div className="archive-table">
            <div className="archive-row archive-head">
              <span>File Name</span>
              <span>Type</span>
              <span>AI Model</span>
              <span>Completion</span>
              <span>Actions</span>
            </div>
            {items.map((item) => (
              <div className="archive-row" key={`${item.type}-${item.id}`}>
                <div className="archive-name">
                  <span className="media-thumb small">
                    {item.type === "tts" ? "TT" : "AV"}
                  </span>
                  <div>
                    <strong>{item.name}</strong>
                    <small>{item.detail}</small>
                  </div>
                </div>
                <span className="capitalize">{item.type}</span>
                <span className="badge">{item.model}</span>
                <span>{formatDate(item.createdAt)}</span>
                <div className="archive-actions">
                  {item.type === "transcription" && (
                    <>
                      <Link
                        className="archive-link-button"
                        to={`/editor/${item.id}`}
                      >
                        Open
                      </Link>
                      <Link
                        className="archive-link-button"
                        to={`/studio/${item.id}`}
                      >
                        Studio
                      </Link>
                    </>
                  )}
                  {item.hasSubtitles && (
                    <>
                      <button
                        type="button"
                        onClick={() => handleSubtitleDownload(item, "srt")}
                      >
                        SRT
                      </button>
                      <button
                        type="button"
                        onClick={() => handleSubtitleDownload(item, "vtt")}
                      >
                        VTT
                      </button>
                    </>
                  )}
                  <button
                    className="danger"
                    type="button"
                    onClick={() => handleDelete(item)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
