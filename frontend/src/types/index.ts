/**
 * API types for AI Transcription & TTS
 */

// STT Models
export interface STTModel {
  id: string;
  name: string;
  type: "whisper" | "parakeet";
  description: string;
}

export interface STTModelsResponse {
  models: STTModel[];
  default_model: string;
  default_whisper: string;
  default_parakeet?: string;
}

// Translation Models
export interface TranslationLanguage {
  code: string;
  name: string;
  nllb_code: string;
  tts_supported: boolean;
}

export interface TranslationModel {
  id: string;
  name: string;
  description: string;
  device: string;
  compute_type: string;
  languages: TranslationLanguage[];
}

export interface TranslationModelsResponse {
  models: TranslationModel[];
  default_model: string;
  default_source_language: string;
  default_target_language: string;
}

// TTS Models
export interface TTSModel {
  id: string;
  name: string;
  description: string;
  sample_rate: number;
  languages: string[];
  model_family?: "supertonic";
  supports_instructions?: boolean;
  supports_voice_presets?: boolean;
  requires_reference_audio?: boolean;
  features?: string[];
}

export interface TTSVoice {
  id: string;
  name: string;
  language: string;
  model_family?: "supertonic" | "all";
  description?: string;
  native_language?: string;
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
  model_type: "whisper" | "parakeet";
}

export interface TranslationResult {
  language: string;
  source_language: string | null;
  model: string;
  text: string;
  segments: TranscriptionSegment[];
  created_at: string;
  tts_audio_path?: string | null;
  tts_voice?: string | null;
  tts_model?: string | null;
  tts_speed?: number | null;
  tts_duration?: number | null;
  tts_sample_rate?: number | null;
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
  model_type: "whisper" | "parakeet";
  is_video: boolean;
}

export interface TranscriptionInfo {
  id: string;
  filename: string;
  result: TranscriptionResult;
  created_at: string;
  is_video: boolean;
  source_size?: number;
  subtitle_paths?: Record<string, string>;
  media_paths?: Record<string, string>;
  translations?: Record<string, TranslationResult>;
  model_used: string;
  model_type: "whisper" | "parakeet";
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
  instruction?: string | null;
  output_format?: "wav" | "mp3";
}

export interface TTSCacheEntry {
  id: string;
  text: string;
  model: string;
  voice: string;
  speed: number;
  pitch: number;
  language: string | null;
  instruction?: string | null;
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
    default_parakeet?: string;
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
