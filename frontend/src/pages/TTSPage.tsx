/**
 * Text-to-Speech Page
 */
import { useState, useEffect } from 'react';
import * as api from '../api/client';
import { TTSModel, TTSVoice } from '../types';

export function TTSPage() {
  const [text, setText] = useState('');
  const [model, setModel] = useState('qwen3-tts-1.8b');
  const [voice, setVoice] = useState('default');
  const [speed, setSpeed] = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [language, setLanguage] = useState('');
  const [outputFormat, setOutputFormat] = useState<'wav' | 'mp3'>('wav');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);

  const [availableVoices, setAvailableVoices] = useState<TTSVoice[]>([]);

  useEffect(() => {
    api.getTTSVoices()
      .then((res) => setAvailableVoices(res.voices))
      .catch(console.error);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!text.trim()) {
      setError('Please enter text to synthesize');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const { audioBlob, duration: audioDuration } = await api.synthesizeSpeech(text, {
        model,
        voice,
        speed,
        pitch,
        language: language || null,
        output_format: outputFormat,
      });

      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      setDuration(audioDuration);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'TTS synthesis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setText('');
    setAudioUrl(null);
    setDuration(null);
    setError(null);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Text-to-Speech</h1>
        <p>Convert text to natural-sounding speech using AI voice synthesis</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="text">Text to Synthesize</label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the text you want to convert to speech..."
              maxLength={5000}
              disabled={loading}
              rows={6}
            />
            <div className="char-count">
              {text.length} / 5000 characters
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="model">TTS Model</label>
              <select
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={loading}
              >
                <optgroup label="Qwen3-TTS Models">
                  <option value="qwen3-tts-0.6b">Qwen3-TTS 0.6B (Fast)</option>
                  <option value="qwen3-tts-1.8b">Qwen3-TTS 1.8B (Balanced)</option>
                  <option value="qwen3-tts-4b">Qwen3-TTS 4B (Best)</option>
                </optgroup>
                <optgroup label="CosyVoice Models">
                  <option value="cosyvoice-300m">CosyVoice 300M</option>
                  <option value="cosyvoice-300m-sft">CosyVoice 300M SFT</option>
                  <option value="cosyvoice-300m-instruct">CosyVoice 300M Instruct</option>
                </optgroup>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="voice">Voice</label>
              <select
                id="voice"
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
                disabled={loading}
              >
                {availableVoices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} ({v.language})
                  </option>
                ))}
              </select>
            </div>

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
                <option value="zh">Chinese</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Speed: {speed.toFixed(1)}x</label>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                disabled={loading}
              />
              <div className="range-labels">
                <span>0.5x</span>
                <span>2.0x</span>
              </div>
            </div>

            <div className="form-group">
              <label>Pitch: {pitch.toFixed(1)}</label>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={pitch}
                onChange={(e) => setPitch(parseFloat(e.target.value))}
                disabled={loading}
              />
              <div className="range-labels">
                <span>Low</span>
                <span>High</span>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="format">Format</label>
              <select
                id="format"
                value={outputFormat}
                onChange={(e) => setOutputFormat(e.target.value as 'wav' | 'mp3')}
                disabled={loading}
              >
                <option value="wav">WAV (Uncompressed)</option>
                <option value="mp3">MP3 (Compressed)</option>
              </select>
            </div>
          </div>

          {error && (
            <div className="alert alert-error">
              <span>⚠️</span> {error}
            </div>
          )}

          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={loading || !text.trim()}>
              {loading ? 'Synthesizing...' : 'Synthesize'}
            </button>
            {audioUrl && (
              <button type="button" className="btn btn-secondary" onClick={handleReset}>
                Clear
              </button>
            )}
          </div>
        </form>

        {audioUrl && (
          <div className="result-section">
            <div className="result-header">
              <h3>Generated Audio</h3>
              {duration && (
                <span className="badge">Duration: {duration.toFixed(2)}s</span>
              )}
            </div>

            <div className="audio-container">
              <audio controls src={audioUrl} autoPlay className="audio-player" />
            </div>

            <div className="result-actions">
              <a
                href={audioUrl}
                download={`tts_${Date.now()}.${outputFormat}`}
                className="btn btn-outline"
              >
                💾 Download Audio
              </a>
            </div>
          </div>
        )}
      </div>

      <div className="info-cards">
        <div className="info-card">
          <h4>Voice Options</h4>
          <p>Multiple voices with different tones</p>
        </div>
        <div className="info-card">
          <h4>Speed Control</h4>
          <p>Adjust playback speed (0.5x - 2.0x)</p>
        </div>
        <div className="info-card">
          <h4>Pitch Control</h4>
          <p>Modify voice pitch</p>
        </div>
      </div>
    </div>
  );
}