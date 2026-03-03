/**
 * Transcription Page
 */
import { useState, useRef } from 'react';
import * as api from '../api/client';
import { TranscriptionResponse } from '../types';

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.mp4', '.mov', '.mkv', '.flac', '.ogg', '.webm', '.m4a'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export function TranscriptionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('');
  const [model, setModel] = useState('whisper-base');
  const [task, setTask] = useState<'transcribe' | 'translate'>('transcribe');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<TranscriptionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const ext = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        setError(`Invalid file format. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`);
        return;
      }
      if (selectedFile.size > MAX_FILE_SIZE) {
        setError('File too large. Maximum size is 50MB.');
        return;
      }
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + Math.random() * 15, 90));
    }, 300);

    try {
      const response = await api.transcribeFile(file, {
        language: language || undefined,
        model: model || undefined,
        task: task,
      });
      clearInterval(progressInterval);
      setProgress(100);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Transcription failed');
    } finally {
      clearInterval(progressInterval);
      setLoading(false);
    }
  };

  const handleCopyToClipboard = () => {
    if (result?.text) {
      navigator.clipboard.writeText(result.text);
    }
  };

  const handleDownloadTxt = () => {
    if (!result?.text) return;
    const blob = new Blob([result.text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadSubtitle = async (format: 'srt' | 'vtt') => {
    if (!result?.transcription_id) return;
    try {
      const { content, filename } = await api.downloadSubtitle(result.transcription_id, format);
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
      setError(err instanceof Error ? err.message : 'Failed to download subtitle');
    }
  };

  const handleReset = () => {
    setFile(null);
    setLanguage('');
    setResult(null);
    setError(null);
    setProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const wordCount = result?.text ? result.text.trim().split(/\s+/).filter((w) => w).length : 0;

  return (
    <div className="page">
      <div className="page-header">
        <h1>Audio/Video Transcription</h1>
        <p>Convert audio and video files to text using AI-powered speech recognition</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="file">Select Audio or Video File</label>
            <input
              ref={fileInputRef}
              type="file"
              id="file"
              accept={ALLOWED_EXTENSIONS.join(',')}
              onChange={handleFileChange}
              disabled={loading}
            />
            {file && (
              <div className="file-info">
                <span>📁</span>
                <span className="name">{file.name}</span>
                <span className="size">{formatFileSize(file.size)}</span>
              </div>
            )}
          </div>

          {loading && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <p className="progress-text">Processing... {Math.round(progress)}%</p>
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
                <option value="zh">Chinese</option>
                <option value="ja">Japanese</option>
                <option value="ko">Korean</option>
                <option value="pt">Portuguese</option>
                <option value="ru">Russian</option>
                <option value="ar">Arabic</option>
                <option value="it">Italian</option>
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
                <optgroup label="Whisper Models">
                  <option value="whisper-tiny">Tiny (Fastest)</option>
                  <option value="whisper-base">Base (Balanced)</option>
                  <option value="whisper-small">Small</option>
                  <option value="whisper-medium">Medium</option>
                  <option value="whisper-large">Large (Best)</option>
                </optgroup>
                <optgroup label="Qwen3-ASR Models">
                  <option value="qwen3-asr-0.6b">Qwen3-ASR 0.6B</option>
                  <option value="qwen3-asr-1.7b">Qwen3-ASR 1.7B</option>
                </optgroup>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="task">Task</label>
              <select
                id="task"
                value={task}
                onChange={(e) => setTask(e.target.value as 'transcribe' | 'translate')}
                disabled={loading}
              >
                <option value="transcribe">Transcribe</option>
                <option value="translate">Translate to English</option>
              </select>
            </div>
          </div>

          {error && (
            <div className="alert alert-error">
              <span>⚠️</span> {error}
            </div>
          )}

          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={loading || !file}>
              {loading ? 'Processing...' : 'Transcribe'}
            </button>
            {result && (
              <button type="button" className="btn btn-secondary" onClick={handleReset}>
                Clear
              </button>
            )}
          </div>
        </form>

        {result && (
          <div className="result-section">
            <div className="result-header">
              <h3>Result</h3>
              <div className="result-meta">
                <span className="badge">{result.language || 'Auto'}</span>
                <span className="badge badge-secondary">{result.model_used}</span>
              </div>
            </div>

            <div className="result-text">{result.text}</div>

            <div className="result-stats">
              <div className="stat">
                <span className="stat-value">{wordCount.toLocaleString()}</span>
                <span className="stat-label">Words</span>
              </div>
              <div className="stat">
                <span className="stat-value">{result.time_taken}s</span>
                <span className="stat-label">Time</span>
              </div>
            </div>

            {result.segments && result.segments.length > 0 && (
              <div className="result-actions">
                <button className="btn btn-outline" onClick={handleCopyToClipboard}>
                  📋 Copy Text
                </button>
                <button className="btn btn-outline" onClick={handleDownloadTxt}>
                  📄 Download TXT
                </button>
                <button className="btn btn-outline" onClick={() => handleDownloadSubtitle('srt')}>
                  🎬 Download SRT
                </button>
                <button className="btn btn-outline" onClick={() => handleDownloadSubtitle('vtt')}>
                  🎬 Download VTT
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="info-cards">
        <div className="info-card">
          <h4>Supported Audio</h4>
          <p>MP3, WAV, FLAC, OGG, M4A, AAC</p>
        </div>
        <div className="info-card">
          <h4>Supported Video</h4>
          <p>MP4, MOV, MKV, WEBM, AVI</p>
        </div>
        <div className="info-card">
          <h4>Max File Size</h4>
          <p>50 MB</p>
        </div>
      </div>
    </div>
  );
}