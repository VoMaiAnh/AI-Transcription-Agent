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

const getArchivePageSize = () => {
  if (typeof window === "undefined") {
    return 8;
  }

  const rowHeight = window.innerWidth <= 820 ? 98 : 72;
  const reservedHeight = window.innerWidth <= 820 ? 460 : 380;
  const visibleRows = Math.floor(
    (window.innerHeight - reservedHeight) / rowHeight,
  );
  const minRows = window.innerWidth <= 820 ? 3 : 5;
  const maxRows = window.innerWidth <= 1180 ? 9 : 14;

  return Math.min(maxRows, Math.max(minRows, visibleRows));
};

export function ArchivePage() {
  const [transcriptions, setTranscriptions] = useState<TranscriptionInfo[]>([]);
  const [ttsResults, setTtsResults] = useState<TTSCacheEntry[]>([]);
  const [filter, setFilter] = useState<ArchiveType>("all");
  const [query, setQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(getArchivePageSize);
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

  useEffect(() => {
    const handleResize = () => {
      setPageSize(getArchivePageSize());
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
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

  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const safeCurrentPage = Math.min(currentPage, totalPages);
  const pageStartIndex = (safeCurrentPage - 1) * pageSize;
  const pagedItems = items.slice(pageStartIndex, pageStartIndex + pageSize);

  useEffect(() => {
    setCurrentPage(1);
  }, [filter, query]);

  useEffect(() => {
    setCurrentPage((page) => Math.min(page, totalPages));
  }, [totalPages]);

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
            {pagedItems.map((item) => (
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
            <div className="archive-pagination">
              <span>
                Showing {pageStartIndex + 1}-
                {Math.min(pageStartIndex + pagedItems.length, items.length)} of{" "}
                {items.length}
              </span>
              <div className="pagination-controls" aria-label="Archive pages">
                <button
                  type="button"
                  onClick={() =>
                    setCurrentPage((page) => Math.max(1, page - 1))
                  }
                  disabled={safeCurrentPage === 1}
                >
                  Previous
                </button>
                <strong>
                  Page {safeCurrentPage} / {totalPages}
                </strong>
                <button
                  type="button"
                  onClick={() =>
                    setCurrentPage((page) => Math.min(totalPages, page + 1))
                  }
                  disabled={safeCurrentPage === totalPages}
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
