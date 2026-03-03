/**
 * History Page
 */
import { useState, useEffect } from 'react';
import * as api from '../api/client';
import { TranscriptionInfo, TTSCacheEntry } from '../types';

type FilterType = 'all' | 'transcription' | 'tts';

export function HistoryPage() {
  const [filter, setFilter] = useState<FilterType>('all');
  const [transcriptions, setTranscriptions] = useState<TranscriptionInfo[]>([]);
  const [ttsResults, setTtsResults] = useState<TTSCacheEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const [transcriptionsData, ttsData] = await Promise.all([
        api.listTranscriptions().catch(() => ({ transcriptions: [], total: 0 })),
        api.listTTSResults().catch(() => ({ results: [], total: 0 })),
      ]);

      setTranscriptions(transcriptionsData.transcriptions);
      setTtsResults(ttsData.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteTranscription = async (id: string) => {
    if (!confirm('Delete this transcription?')) return;

    try {
      await api.deleteTranscription(id);
      setTranscriptions((prev) => prev.filter((t) => t.id !== id));
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete');
    }
  };

  const handleDeleteTTS = async (id: string) => {
    if (!confirm('Delete this TTS result?')) return;

    try {
      await api.deleteTTSResult(id);
      setTtsResults((prev) => prev.filter((t) => t.id !== id));
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete');
    }
  };

  const handleDownloadSubtitle = async (transcriptionId: string, format: 'srt' | 'vtt') => {
    try {
      const { content, filename } = await api.downloadSubtitle(transcriptionId, format);
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to download subtitle');
    }
  };

  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleString();
  };

  const filteredTranscriptions = filter === 'tts' ? [] : transcriptions;
  const filteredTTS = filter === 'transcription' ? [] : ttsResults;

  const totalTranscriptions = transcriptions.length;
  const totalTTS = ttsResults.length;
  const totalAll = totalTranscriptions + totalTTS;

  return (
    <div className="page">
      <div className="page-header">
        <h1>History</h1>
        <p>View and manage your previous transcriptions and TTS results</p>
      </div>

      <div className="filter-tabs">
        <button
          className={`filter-tab ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          All ({totalAll})
        </button>
        <button
          className={`filter-tab ${filter === 'transcription' ? 'active' : ''}`}
          onClick={() => setFilter('transcription')}
        >
          Transcriptions ({totalTranscriptions})
        </button>
        <button
          className={`filter-tab ${filter === 'tts' ? 'active' : ''}`}
          onClick={() => setFilter('tts')}
        >
          TTS ({totalTTS})
        </button>
      </div>

      <div className="card">
        {loading ? (
          <div className="loading-state">
            <div className="spinner" />
            <p>Loading history...</p>
          </div>
        ) : error ? (
          <div className="alert alert-error">
            <span>⚠️</span> {error}
          </div>
        ) : filteredTranscriptions.length + filteredTTS.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📭</div>
            <h3>No history yet</h3>
            <p>
              {filter === 'all'
                ? 'Transcriptions and TTS results will appear here'
                : filter === 'transcription'
                ? 'Transcriptions will appear here'
                : 'TTS results will appear here'}
            </p>
          </div>
        ) : (
          <div className="history-list">
            {filteredTranscriptions.map((item) => (
              <div key={item.id} className="history-item">
                <div className="history-item-header">
                  <div className="history-item-info">
                    <h4>{item.filename}</h4>
                    <span className="history-item-date">{formatDate(item.created_at)}</span>
                  </div>
                  <div className="history-item-badges">
                    <span className="badge">{item.result.language || 'Auto'}</span>
                    <span className="badge badge-secondary">{item.model_used}</span>
                  </div>
                </div>
                <p className="history-item-excerpt">
                  {item.result.text.substring(0, 150)}
                  {item.result.text.length > 150 ? '...' : ''}
                </p>
                <div className="history-item-meta">
                  <span>📝 {item.result.text.split(/\s+/).length} words</span>
                  <span>⏱️ {item.time_taken}s</span>
                </div>
                <div className="history-item-actions">
                  {item.result.segments && item.result.segments.length > 0 && (
                    <>
                      <button
                        className="btn btn-small btn-outline"
                        onClick={() => handleDownloadSubtitle(item.id, 'srt')}
                      >
                        SRT
                      </button>
                      <button
                        className="btn btn-small btn-outline"
                        onClick={() => handleDownloadSubtitle(item.id, 'vtt')}
                      >
                        VTT
                      </button>
                    </>
                  )}
                  <button
                    className="btn btn-small btn-danger"
                    onClick={() => handleDeleteTranscription(item.id)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}

            {filteredTTS.map((item) => (
              <div key={item.id} className="history-item">
                <div className="history-item-header">
                  <div className="history-item-info">
                    <h4>TTS: {item.model}</h4>
                    <span className="history-item-date">{formatDate(item.created_at)}</span>
                  </div>
                  <div className="history-item-badges">
                    <span className="badge">{item.voice}</span>
                    <span className="badge badge-secondary">{item.duration.toFixed(2)}s</span>
                  </div>
                </div>
                <p className="history-item-excerpt">
                  {item.text.substring(0, 150)}
                  {item.text.length > 150 ? '...' : ''}
                </p>
                <div className="history-item-meta">
                  <span>🎵 Speed: {item.speed}x</span>
                  <span>🎤 Pitch: {item.pitch}</span>
                </div>
                <div className="history-item-actions">
                  <button
                    className="btn btn-small btn-danger"
                    onClick={() => handleDeleteTTS(item.id)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="card-actions">
          <button className="btn btn-secondary" onClick={loadHistory}>
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
}