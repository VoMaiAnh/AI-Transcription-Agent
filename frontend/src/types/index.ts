/**
 * API types for AI Transcription & TTS
 */

// STT Models
export interface STTModel {
  id: string;
  name: string;
  type: 'whisper' | 'qwen3-asr';
  description: string;
}

export interface STTModelsResponse {
  models: STTModel[];
  default_model: string;
  default_whisper: string;
  default_qwen3_asr: string;
}

// TTS Models
export interface TTSModel {
  id: string;
  name: string;
  description: string;
  sample_rate: number;
  languages: string[];
}

export interface TTSVoice {
  id: string;
  name: string;
  language: string;
}

export interface TTSModelsResponse {
  models: TTSModel[];
  default_model: string;
}

export interface TTSVoicesResponse {
  voices: TTSVoice[];
  default_voice: string;
}

// Transcription
export interface TranscriptionSegment {
  id: number;
  start: number;
  end: number;
  text: string;
}

export interface TranscriptionResult {
  text: string;
  language: string | null;
  segments: TranscriptionSegment[];
  model_type: 'whisper' | 'qwen3-asr';
}

export interface TranscriptionResponse {
  success: boolean;
  transcription_id: string;
  filename: string;
  language: string | null;
  text: string;
  segments: TranscriptionSegment[];
  time_taken: number;
  model_used: string;
  model_type: 'whisper' | 'qwen3-asr';
}

export interface TranscriptionInfo {
  id: string;
  filename: string;
  result: TranscriptionResult;
  created_at: string;
  is_video: boolean;
  model_used: string;
  model_type: 'whisper' | 'qwen3-asr';
  time_taken: number;
}

// TTS
export interface TTSRequest {
  text: string;
  model?: string;
  voice?: string;
  speed?: number;
  pitch?: number;
  language?: string | null;
  output_format?: 'wav' | 'mp3';
}

export interface TTSCacheEntry {
  id: string;
  text: string;
  model: string;
  voice: string;
  speed: number;
  pitch: number;
  language: string | null;
  duration: number;
  sample_rate: number;
  created_at: string;
}

// Health
export interface HealthResponse {
  status: string;
  device: string;
  app: {
    name: string;
    version: string;
  };
  stt: {
    default_whisper: string;
    default_qwen3_asr: string;
    available_models: string[];
  };
  tts: {
    available_models: string[];
  };
}

// API Error
export interface ApiError {
  detail: string;
}
