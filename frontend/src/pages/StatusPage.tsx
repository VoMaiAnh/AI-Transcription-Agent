import { useEffect, useState } from "react";
import * as api from "../api/client";
import { HealthResponse, TranslationModelsResponse } from "../types";

export function StatusPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [translationInfo, setTranslationInfo] =
    useState<TranslationModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [translationError, setTranslationError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    api
      .getHealth()
      .then((data) => {
        if (active) {
          setHealth(data);
        }
      })
      .catch((err) => {
        if (active) {
          setError(
            err instanceof Error ? err.message : "Failed to load AI status",
          );
        }
      });

    api
      .getTranslationModels()
      .then((data) => {
        if (active) {
          setTranslationInfo(data);
        }
      })
      .catch((err) => {
        if (active) {
          setTranslationError(
            err instanceof Error
              ? err.message
              : "Failed to load translation models",
          );
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const translationModel = translationInfo?.models[0] || null;
  const translationLanguages = translationModel?.languages || [];
  const visibleTranslationLanguages = translationLanguages.slice(0, 12);
  const extraTranslationLanguageCount =
    translationLanguages.length - visibleTranslationLanguages.length;
  const ttsReadyLanguageCount = translationLanguages.filter(
    (language) => language.tts_supported,
  ).length;

  return (
    <div className="status-page">
      <section className="page-heading">
        <div>
          <h1>AI Status</h1>
          <p>Current backend health and available AI model inventory.</p>
        </div>
      </section>

      {error && <div className="alert-card">{error}</div>}

      <section className="status-grid">
        <div className="glass-card">
          <h2>Runtime</h2>
          <div className="metric-row">
            <span>Status</span>
            <strong>{health?.status || "Unknown"}</strong>
          </div>
          <div className="metric-row">
            <span>Device</span>
            <strong>{health?.device || "Unknown"}</strong>
          </div>
          <div className="metric-row">
            <span>Application</span>
            <strong>
              {health ? `${health.app.name} ${health.app.version}` : "Loading"}
            </strong>
          </div>
        </div>

        <div className="glass-card">
          <h2>Speech-to-Text</h2>
          <p>Default Whisper: {health?.stt.default_whisper || "Loading"}</p>
          <p>Default Parakeet: {health?.stt.default_parakeet || "Loading"}</p>
          <div className="tag-list">
            {(health?.stt.available_models || []).map((model) => (
              <span className="badge" key={model}>
                {model}
              </span>
            ))}
          </div>
        </div>

        <div className="glass-card">
          <h2>Text-to-Speech</h2>
          <div className="tag-list">
            {(health?.tts.available_models || []).map((model) => (
              <span className="badge" key={model}>
                {model}
              </span>
            ))}
          </div>
        </div>

        <div className="glass-card translation-status-card">
          <h2>Translation</h2>
          {translationError ? (
            <div className="alert-card">{translationError}</div>
          ) : translationModel ? (
            <>
              <div className="metric-row">
                <span>Default Model</span>
                <strong>{translationInfo?.default_model}</strong>
              </div>
              <div className="metric-row">
                <span>Runtime</span>
                <strong>
                  {translationModel.device.toUpperCase()} /{" "}
                  {translationModel.compute_type}
                </strong>
              </div>
              <div className="metric-row">
                <span>Languages</span>
                <strong>
                  {translationLanguages.length} total / {ttsReadyLanguageCount}{" "}
                  TTS-ready
                </strong>
              </div>
              <div className="model-note">
                <strong>{translationModel.name}</strong>
                <span>{translationModel.description}</span>
              </div>
              <div className="tag-list">
                {visibleTranslationLanguages.map((language) => (
                  <span
                    className="badge"
                    key={language.code}
                    title={language.nllb_code}
                  >
                    {language.name} ({language.code})
                  </span>
                ))}
                {extraTranslationLanguageCount > 0 && (
                  <span className="badge">
                    +{extraTranslationLanguageCount} more
                  </span>
                )}
              </div>
            </>
          ) : (
            <p>Loading translation models...</p>
          )}
        </div>
      </section>
    </div>
  );
}
