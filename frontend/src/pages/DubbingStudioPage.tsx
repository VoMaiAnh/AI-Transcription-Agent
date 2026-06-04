import { FormEvent, useRef, useState } from 'react';
import * as api from '../api/client';
import { TranscriptionResponse } from '../types';
import {
  ALLOWED_MEDIA_EXTENSIONS,
  downloadBlob,
  formatFileSize,
  formatTime,
  validateMediaFile,
} from './pageUtils';

const SUPERTONIC_MODEL = { id: 'Supertone/supertonic-3', name: 'Supertonic 3' };

const SUPERTONIC_LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Spanish' },
  { code: 'fr', label: 'French' },
  { code: 'de', label: 'German' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'ru', label: 'Russian' },
  { code: 'ar', label: 'Arabic' },
  { code: 'hi', label: 'Hindi' },
  { code: 'it', label: 'Italian' },
  { code: 'nl', label: 'Dutch' },
  { code: 'pl', label: 'Polish' },
  { code: 'tr', label: 'Turkish' },
  { code: 'uk', label: 'Ukrainian' },
  { code: 'vi', label: 'Vietnamese' },
  { code: 'na', label: 'Unknown / fallback' },
];

const SUPERTONIC_VOICES = [
  { id: 'M1', name: 'Supertonic M1' },
  { id: 'M2', name: 'Supertonic M2' },
  { id: 'M3', name: 'Supertonic M3' },
  { id: 'M4', name: 'Supertonic M4' },
  { id: 'M5', name: 'Supertonic M5' },
  { id: 'F1', name: 'Supertonic F1' },
  { id: 'F2', name: 'Supertonic F2' },
  { id: 'F3', name: 'Supertonic F3' },
  { id: 'F4', name: 'Supertonic F4' },
  { id: 'F5', name: 'Supertonic F5' },
];

export function DubbingStudioPage() {
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('');
  const [subtitleModel, setSubtitleModel] = useState('parakeet-tdt-0.6b');
  const [result, setResult] = useState<TranscriptionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [mediaLoading, setMediaLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [subtitleFormat, setSubtitleFormat] = useState<'srt' | 'vtt'>('srt');
  const [preview, setPreview] = useState<string | null>(null);
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [voice, setVoice] = useState('M1');
  const [ttsModel, setTtsModel] = useState(SUPERTONIC_MODEL.id);
  const [speed, setSpeed] = useState(1);
  const [originalVolume, setOriginalVolume] = useState(0.15);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    const validationError = validateMediaFile(selectedFile);
    if (validationError) {
      setError(validationError);
      return;
    }

    setFile(selectedFile);
    setError(null);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError('Please select a media file before opening the dubbing timeline.');
      return;
    }

    setLoading(true);
    setError(null);
    setPreview(null);

    try {
      const response = await api.transcribeFile(file, {
        language: language || undefined,
        model: subtitleModel,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to prepare dubbing timeline');
    } finally {
      setLoading(false);
    }
  };

  const handlePreview = async () => {
    if (!result) return;
    try {
      const { content } = await api.downloadSubtitle(result.transcription_id, subtitleFormat);
      setPreview(content);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load subtitle preview');
    }
  };

  const handleDownloadSubtitle = async () => {
    if (!result) return;
    try {
      const { content, filename, mediaType } = await api.downloadSubtitle(result.transcription_id, subtitleFormat);
      downloadBlob(new Blob([content], { type: mediaType }), filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download subtitle track');
    }
  };

  const handleEmbedVideo = async (mode: 'soft' | 'hard') => {
    if (!result) return;
    setMediaLoading(mode);
    setError(null);

    try {
      const { blob, filename } = await api.embedSubtitleVideo(result.transcription_id, {
        mode,
        format: subtitleFormat,
      });
      downloadBlob(blob, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to render subtitled video');
    } finally {
      setMediaLoading(null);
    }
  };

  const handleDubVideo = async () => {
    if (!result) return;
    setMediaLoading('dub');
    setError(null);

    try {
      const { blob, filename } = await api.dubVideo(result.transcription_id, {
        target_language: targetLanguage,
        tts_model: ttsModel,
        voice,
        speed,
        pitch: 1,
        original_volume: originalVolume,
        whisper_model: subtitleModel,
      });
      downloadBlob(blob, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to render final dub');
    } finally {
      setMediaLoading(null);
    }
  };

  const segments = result?.segments || [];
  const canExportVideo = Boolean(result?.is_video && segments.length > 0);

  return (
    <div className="studio-grid dubbing-grid">
      <aside className="glass-card voice-library">
        <h2>Voice Library</h2>
        <div className="model-note">
          <strong>Supertonic 3</strong>
          <span>Built-in voice styles for fast on-device dubbing.</span>
        </div>
        {SUPERTONIC_VOICES.map((item) => (
          <button
            key={item.id}
            className={`voice-card ${voice === item.id ? 'active' : ''}`}
            type="button"
            onClick={() => setVoice(item.id)}
          >
            <span className="avatar-token">{item.id}</span>
            <span>{item.name}</span>
          </button>
        ))}
      </aside>

      <section className="stage-column">
        <form className="glass-card media-workbench" onSubmit={handleSubmit}>
          <div className="card-header">
            <div>
              <p className="eyebrow">Timeline Master</p>
              <h1>Dubbing Studio</h1>
            </div>
            <span className="badge">{result ? 'Timeline Ready' : 'Prepare Timeline'}</span>
          </div>

          <label className="drop-zone compact">
            <input
              ref={fileInputRef}
              type="file"
              accept={ALLOWED_MEDIA_EXTENSIONS.join(',')}
              onChange={handleFileChange}
              disabled={loading}
            />
            <span>{file ? file.name : 'Select source media'}</span>
            <small>{file ? formatFileSize(file.size) : 'Generate transcript timing before export'}</small>
          </label>

          <div className="control-grid">
            <label>
              Source Language
              <select value={language} onChange={(event) => setLanguage(event.target.value)} disabled={loading}>
                <option value="">Auto-detect</option>
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="zh">Chinese</option>
                <option value="ja">Japanese</option>
              </select>
            </label>
            <label>
              Timing Model
              <select value={subtitleModel} onChange={(event) => setSubtitleModel(event.target.value)} disabled={loading}>
                <option value="parakeet-tdt-0.6b">Parakeet TDT 0.6B</option>
                <option value="whisper-base">Whisper Base</option>
                <option value="whisper-medium">Whisper Medium</option>
                <option value="whisper-large">Whisper Large</option>
              </select>
            </label>
          </div>

          {error && <div className="alert-card">{error}</div>}

          <div className="button-row">
            <button className="studio-button primary" disabled={loading || !file} type="submit">
              {loading ? 'Preparing' : 'Build Timeline'}
            </button>
            {result && (
              <button
                className="studio-button ghost"
                type="button"
                onClick={() => {
                  setResult(null);
                  setPreview(null);
                  setFile(null);
                  if (fileInputRef.current) fileInputRef.current.value = '';
                }}
              >
                Clear
              </button>
            )}
          </div>
        </form>

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
            <span>{file?.name || 'Source media track'}</span>
          </div>
          <div className="timeline-track audio-track">
            {Array.from({ length: 48 }).map((_, index) => (
              <i key={index} style={{ height: `${15 + ((index * 13) % 65)}%` }} />
            ))}
          </div>
          <div className="timeline-track dub-track">
            <span>{segments.length ? `Dubbing: ${voice}` : 'Dubbed audio track'}</span>
          </div>
          <div className="segment-strip">
            {segments.slice(0, 5).map((segment) => (
              <span key={segment.id}>{formatTime(segment.start)}</span>
            ))}
          </div>
        </section>

        {preview && (
          <section className="glass-card preview-card">
            <div className="card-header">
              <h2>{subtitleFormat.toUpperCase()} Preview</h2>
              <button className="studio-button ghost small" type="button" onClick={() => navigator.clipboard.writeText(preview)}>
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
            Target Language
            <select value={targetLanguage} onChange={(event) => setTargetLanguage(event.target.value)}>
              {SUPERTONIC_LANGUAGES.map((item) => (
                <option key={item.code} value={item.code}>{item.label}</option>
              ))}
            </select>
          </label>
          <label>
            TTS Model
            <select value={ttsModel} onChange={(event) => setTtsModel(event.target.value)}>
              <option value={SUPERTONIC_MODEL.id}>{SUPERTONIC_MODEL.name}</option>
            </select>
          </label>
          <label>
            Voice
            <select value={voice} onChange={(event) => setVoice(event.target.value)}>
              {SUPERTONIC_VOICES.map((item) => (
                <option key={item.id} value={item.id}>{item.name}</option>
              ))}
            </select>
          </label>
          <label className="range-control">
            <span>Speed <strong>{speed.toFixed(2)}x</strong></span>
            <input min="0.7" max="2" step="0.05" type="range" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
          </label>
          <label className="range-control">
            <span>Original Volume <strong>{Math.round(originalVolume * 100)}%</strong></span>
            <input min="0" max="1" step="0.05" type="range" value={originalVolume} onChange={(event) => setOriginalVolume(Number(event.target.value))} />
          </label>
        </section>

        <section className="glass-card export-card">
          <h2>Export & Deliver</h2>
          <label>
            Subtitle Format
            <select value={subtitleFormat} onChange={(event) => setSubtitleFormat(event.target.value as 'srt' | 'vtt')}>
              <option value="srt">SRT</option>
              <option value="vtt">VTT</option>
            </select>
          </label>
          <button className="export-option" type="button" disabled={!result} onClick={handlePreview}>
            <strong>Preview Subtitle Tracks</strong>
            <span>SRT/VTT transcript timing</span>
          </button>
          <button className="export-option" type="button" disabled={!result} onClick={handleDownloadSubtitle}>
            <strong>Download Subtitle Tracks</strong>
            <span>Deliver standalone captions</span>
          </button>
          <button className="export-option" type="button" disabled={!canExportVideo || mediaLoading !== null} onClick={() => handleEmbedVideo('soft')}>
            <strong>{mediaLoading === 'soft' ? 'Rendering...' : 'Embed Soft Subtitles'}</strong>
            <span>Multilingual MKV stream</span>
          </button>
          <button className="export-option" type="button" disabled={!canExportVideo || mediaLoading !== null} onClick={() => handleEmbedVideo('hard')}>
            <strong>{mediaLoading === 'hard' ? 'Rendering...' : 'Hard Burn Subtitles'}</strong>
            <span>Permanent MP4 render</span>
          </button>
          <button className="studio-button primary full" type="button" disabled={!canExportVideo || mediaLoading !== null} onClick={handleDubVideo}>
            {mediaLoading === 'dub' ? 'Rendering Final Dub' : 'Render Final Dub'}
          </button>
        </section>
      </aside>
    </div>
  );
}
