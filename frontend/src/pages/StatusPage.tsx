import { useEffect, useState } from 'react';
import * as api from '../api/client';
import { HealthResponse } from '../types';

export function StatusPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getHealth()
      .then(setHealth)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load AI status'));
  }, []);

  return (
    <div className="status-page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">Model Telemetry</p>
          <h1>AI Status</h1>
          <p>Current backend health and available speech models.</p>
        </div>
      </section>

      {error && <div className="alert-card">{error}</div>}

      <section className="status-grid">
        <div className="glass-card">
          <h2>Runtime</h2>
          <div className="metric-row">
            <span>Status</span>
            <strong>{health?.status || 'Unknown'}</strong>
          </div>
          <div className="metric-row">
            <span>Device</span>
            <strong>{health?.device || 'Unknown'}</strong>
          </div>
          <div className="metric-row">
            <span>Application</span>
            <strong>{health ? `${health.app.name} ${health.app.version}` : 'Loading'}</strong>
          </div>
        </div>

        <div className="glass-card">
          <h2>Speech-to-Text</h2>
          <p>Default Whisper: {health?.stt.default_whisper || 'Loading'}</p>
          <p>Default Parakeet: {health?.stt.default_parakeet || 'Loading'}</p>
          <div className="tag-list">
            {(health?.stt.available_models || []).map((model) => (
              <span className="badge" key={model}>{model}</span>
            ))}
          </div>
        </div>

        <div className="glass-card">
          <h2>Text-to-Speech</h2>
          <div className="tag-list">
            {(health?.tts.available_models || []).map((model) => (
              <span className="badge" key={model}>{model}</span>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
