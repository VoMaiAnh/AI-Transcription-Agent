import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import * as api from '../api/client';
import { HealthResponse, TranscriptionInfo, TTSCacheEntry } from '../types';
import { formatDate } from './pageUtils';

type DashboardItem = {
  id: string;
  name: string;
  type: 'Transcription' | 'TTS';
  model: string;
  createdAt: string;
  language?: string | null;
};

export function DashboardPage() {
  const [transcriptions, setTranscriptions] = useState<TranscriptionInfo[]>([]);
  const [ttsResults, setTtsResults] = useState<TTSCacheEntry[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function loadDashboard() {
      setLoading(true);
      setError(null);

      try {
        const [transcriptionData, ttsData, healthData] = await Promise.all([
          api.listTranscriptions(),
          api.listTTSResults(),
          api.getHealth(),
        ]);

        if (!mounted) return;
        setTranscriptions(transcriptionData.transcriptions);
        setTtsResults(ttsData.results);
        setHealth(healthData);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
      } finally {
        if (mounted) setLoading(false);
      }
    }

    loadDashboard();
    return () => {
      mounted = false;
    };
  }, []);

  const recentItems = useMemo<DashboardItem[]>(() => {
    const transcriptionItems = transcriptions.map((item) => ({
      id: item.id,
      name: item.filename,
      type: 'Transcription' as const,
      model: item.model_used,
      createdAt: item.created_at,
      language: item.result.language,
    }));

    const ttsItems = ttsResults.map((item) => ({
      id: item.id,
      name: item.text.slice(0, 70) || `TTS result ${item.id}`,
      type: 'TTS' as const,
      model: item.model,
      createdAt: item.created_at,
      language: item.language,
    }));

    return [...transcriptionItems, ...ttsItems]
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, 6);
  }, [transcriptions, ttsResults]);

  const languageSummary = useMemo(() => {
    const counts = new Map<string, number>();
    transcriptions.forEach((item) => {
      const language = item.result.language || 'Auto';
      counts.set(language, (counts.get(language) || 0) + 1);
    });
    return Array.from(counts.entries()).slice(0, 4);
  }, [transcriptions]);

  return (
    <div className="dashboard-page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">AI Media Studio</p>
          <h1>Project Dashboard</h1>
          <p>Overview of your AI transcription and dubbing workflows.</p>
        </div>
        <div className="glass-card compact-status">
          <span className="status-dot" />
          {health ? `${health.status} on ${health.device}` : 'Checking systems'}
        </div>
      </section>

      {error && <div className="alert-card">{error}</div>}

      <section className="dashboard-grid">
        <div className="glass-card recent-projects">
          <div className="card-header">
            <h2>Recent Projects</h2>
            <Link to="/archive">View All</Link>
          </div>
          {loading ? (
            <div className="loading-panel">Loading project history...</div>
          ) : recentItems.length === 0 ? (
            <div className="empty-panel">
              <strong>No projects yet</strong>
              <span>Start a transcription or dubbing pass to populate the dashboard.</span>
            </div>
          ) : (
            <div className="project-list">
              {recentItems.map((item) => (
                <article className="project-row" key={`${item.type}-${item.id}`}>
                  <div className="media-thumb">{item.type === 'TTS' ? 'TT' : 'AV'}</div>
                  <div>
                    <h3>{item.name}</h3>
                    <p>
                      {item.type} · {item.model} · {item.language || 'Auto'}
                    </p>
                  </div>
                  <div className="row-meta">
                    <span className="badge success">Completed</span>
                    <span>{formatDate(item.createdAt)}</span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>

        <div className="glass-card quick-actions">
          <h2>Quick Actions</h2>
          <Link className="studio-button primary" to="/editor">
            New Transcription
          </Link>
          <Link className="studio-button ghost" to="/studio">
            Start Dubbing
          </Link>
          <Link className="studio-button ghost" to="/archive">
            Open Archive
          </Link>
        </div>

        <div className="glass-card usage-card">
          <h2>Storage & Usage</h2>
          <div className="metric-row">
            <span>Transcriptions</span>
            <strong>{transcriptions.length}</strong>
          </div>
          <div className="meter">
            <span style={{ width: `${Math.min(transcriptions.length * 8, 100)}%` }} />
          </div>
          <div className="metric-row">
            <span>TTS Results</span>
            <strong>{ttsResults.length}</strong>
          </div>
          <div className="meter cyan">
            <span style={{ width: `${Math.min(ttsResults.length * 8, 100)}%` }} />
          </div>
        </div>

        <div className="glass-card queue-card">
          <div className="card-header">
            <h2>AI Processing Queue</h2>
            <span className="badge">{loading ? 'Loading' : 'Ready'}</span>
          </div>
          <div className="queue-item">
            <strong>Speech Recognition</strong>
            <span>{health?.stt.available_models.length || 0} models available</span>
            <div className="meter">
              <span style={{ width: health ? '100%' : '35%' }} />
            </div>
          </div>
          <div className="queue-item">
            <strong>Voice Synthesis</strong>
            <span>{health?.tts.available_models.length || 0} models available</span>
            <div className="meter lime">
              <span style={{ width: health ? '100%' : '25%' }} />
            </div>
          </div>
        </div>

        <div className="glass-card language-card">
          <h2>Language Distribution</h2>
          {languageSummary.length === 0 ? (
            <div className="empty-panel small">No language data yet.</div>
          ) : (
            <div className="language-list">
              {languageSummary.map(([language, count]) => (
                <div className="metric-row" key={language}>
                  <span>{language}</span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
