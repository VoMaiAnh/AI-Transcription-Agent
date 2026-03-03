/**
 * API Client for AI Transcription & TTS Backend
 */

import {
  STTModelsResponse,
  TTSModelsResponse,
  TTSVoicesResponse,
  TranscriptionResponse,
  TranscriptionInfo,
  TTSCacheEntry,
  HealthResponse,
  ApiError,
} from '../types';

const API_BASE = '/api/v1';

/**
 * Handle API response and throw error if not ok
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: 'An unexpected error occurred',
    }));
    throw new Error(error.detail);
  }
  return response.json();
}

/**
 * Health check endpoint
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch('/health');
  return handleResponse<HealthResponse>(response);
}

/**
 * STT Models
 */
export async function getSTTModels(): Promise<STTModelsResponse> {
  const response = await fetch(`${API_BASE}/models`);
  return handleResponse<STTModelsResponse>(response);
}

/**
 * Transcribe audio/video file
 */
export async function transcribeFile(
  file: File,
  options?: {
    language?: string;
    model?: string;
    task?: 'transcribe' | 'translate';
  }
): Promise<TranscriptionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  if (options?.language) {
    formData.append('language', options.language);
  }
  if (options?.model) {
    formData.append('model', options.model);
  }
  if (options?.task) {
    formData.append('task', options.task);
  }

  const response = await fetch(`${API_BASE}/transcribe`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse<TranscriptionResponse>(response);
}

/**
 * Get transcription by ID
 */
export async function getTranscription(
  transcriptionId: string
): Promise<TranscriptionInfo> {
  const response = await fetch(`${API_BASE}/transcription/${transcriptionId}`);
  return handleResponse<TranscriptionInfo>(response);
}

/**
 * Delete transcription by ID
 */
export async function deleteTranscription(
  transcriptionId: string
): Promise<{ message: string; id: string }> {
  const response = await fetch(`${API_BASE}/transcription/${transcriptionId}`, {
    method: 'DELETE',
  });
  return handleResponse<{ message: string; id: string }>(response);
}

/**
 * List all transcriptions
 */
export async function listTranscriptions(): Promise<{
  transcriptions: TranscriptionInfo[];
  total: number;
}> {
  const response = await fetch(`${API_BASE}/list`);
  return handleResponse<{ transcriptions: TranscriptionInfo[]; total: number }>(
    response
  );
}

/**
 * Download subtitle file
 */
export async function downloadSubtitle(
  transcriptionId: string,
  format: 'srt' | 'vtt' = 'srt'
): Promise<{ content: string; filename: string; mediaType: string }> {
  const formData = new FormData();
  formData.append('format', format);

  const response = await fetch(`${API_BASE}/subtitle/${transcriptionId}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: 'Failed to download subtitle',
    }));
    throw new Error(error.detail);
  }

  const content = await response.text();
  const contentDisposition = response.headers.get('Content-Disposition');
  const filename = contentDisposition
    ? contentDisposition.split('filename=')[1]?.replace(/"/g, '') ||
      `subtitle.${format}`
    : `subtitle.${format}`;
  const mediaType = response.headers.get('Content-Type') || 'text/plain';

  return { content, filename, mediaType };
}

/**
 * TTS Models
 */
export async function getTTSModels(): Promise<TTSModelsResponse> {
  const response = await fetch(`${API_BASE}/tts/models`);
  return handleResponse<TTSModelsResponse>(response);
}

/**
 * TTS Voices
 */
export async function getTTSVoices(): Promise<TTSVoicesResponse> {
  const response = await fetch(`${API_BASE}/tts/voices`);
  return handleResponse<TTSVoicesResponse>(response);
}

/**
 * Synthesize speech from text
 */
export async function synthesizeSpeech(
  text: string,
  options?: {
    model?: string;
    voice?: string;
    speed?: number;
    pitch?: number;
    language?: string | null;
    output_format?: 'wav' | 'mp3';
  }
): Promise<{ audioBlob: Blob; ttsId: string; duration: number; sampleRate: number }> {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('model', options?.model || 'qwen3-tts-1.8b');
  formData.append('voice', options?.voice || 'default');
  formData.append('speed', (options?.speed ?? 1.0).toString());
  formData.append('pitch', (options?.pitch ?? 1.0).toString());
  if (options?.language !== undefined) {
    formData.append('language', options.language || '');
  }
  formData.append('output_format', options?.output_format || 'wav');

  const response = await fetch(`${API_BASE}/tts/synthesize`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: 'TTS synthesis failed',
    }));
    throw new Error(error.detail);
  }

  const audioBlob = await response.blob();
  const ttsId = response.headers.get('X-TTS-ID') || '';
  const duration = parseFloat(response.headers.get('X-Duration') || '0');
  const sampleRate = parseInt(response.headers.get('X-Sample-Rate') || '24000');

  return { audioBlob, ttsId, duration, sampleRate };
}

/**
 * Get TTS result by ID
 */
export async function getTTSResult(
  ttsId: string
): Promise<TTSCacheEntry> {
  const response = await fetch(`${API_BASE}/tts/result/${ttsId}`);
  return handleResponse<TTSCacheEntry>(response);
}

/**
 * Delete TTS result by ID
 */
export async function deleteTTSResult(
  ttsId: string
): Promise<{ message: string; id: string }> {
  const response = await fetch(`${API_BASE}/tts/result/${ttsId}`, {
    method: 'DELETE',
  });
  return handleResponse<{ message: string; id: string }>(response);
}

/**
 * List all TTS results
 */
export async function listTTSResults(): Promise<{
  results: TTSCacheEntry[];
  total: number;
}> {
  const response = await fetch(`${API_BASE}/tts/list`);
  return handleResponse<{ results: TTSCacheEntry[]; total: number }>(response);
}
