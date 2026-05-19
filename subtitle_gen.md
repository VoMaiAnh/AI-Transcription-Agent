# NVIDIA Parakeet TDT 0.6B Subtitle Generation

> **Status**: Implementation Complete (2026-04-05)  
> **Current Implementation**: Whisper-based for reliable CPU inference  
> **Future Enhancement**: True Parakeet TDT via NVIDIA NeMo toolkit (GPU)

## Overview

This document describes the subtitle generation feature in the AI Transcription Agent.

**Current Implementation**: Uses Whisper model for reliable subtitle generation with accurate timestamps  
**Target Model**: NVIDIA Parakeet TDT 0.6B v3 (future GPU optimization)

---

## Current Implementation Details

### Architecture

The "Parakeet" subtitle generation feature currently uses **Whisper** as the inference engine for reliable CPU-based subtitle generation. This provides:

- Accurate word-level timestamps
- Support for multiple languages
- Reliable SRT/VTT output
- CPU-efficient inference

### Model Information (Target: Parakeet TDT)

| Property | Value |
|----------|-------|
| **Target Model** | `nvidia/parakeet-tdt-0.6b-v3` |
| **Parameters** | 600M |
| **Architecture** | Transducer (TDT) / FastConformer |
| **License** | CC-BY-4.0 |
| **Model Size** | ~1.2GB |

### Current Model (Whisper)

| Property | Value |
|----------|-------|
| **Model** | Whisper Small |
| **Parameters** | 244M |
| **Architecture** | Transformer Encoder-Decoder |
| **Device** | CPU |

### Supported Languages

**Current (Whisper)**: 99+ languages including English, Spanish, French, German, Italian, Portuguese, Russian, Ukrainian, Chinese, Japanese, Korean, Arabic, Dutch, Polish, and more.

**Target (Parakeet)**: 24+ languages - English, European languages, Russian, Ukrainian

---

## Implementation Status

### Completed

| Component | Status | Description |
|-----------|--------|-------------|
| `requirements.txt` | Done | whisper, soundfile, scipy |
| `config.py` | Done | PARAKEET_MODEL setting added |
| `parakeet_service.py` | Done | Whisper-based inference service |
| `transcription_service.py` | Done | Parakeet model routing |
| `transcription.py` | Done | API endpoint updated |
| Frontend Tab | Done | Dedicated Subtitle Generator page |

### Future Enhancements

| Component | Status | Description |
|-----------|--------|-------------|
| NeMo Integration | Planned | Full Parakeet TDT via NVIDIA NeMo |
| GPU Acceleration | Planned | CUDA support for faster inference |
| ONNX Optimization | Planned | Pre-converted ONNX model |

---

## File Structure

```
AI-Transcription-Agent/
├── app/
│   ├── config.py                          # UPDATED: PARAKEET_MODEL setting
│   ├── services/
│   │   ├── transcription_service.py       # UPDATED: Parakeet routing
│   │   ├── parakeet_service.py            # NEW: Subtitle generation service
│   │   └── subtitle_service.py            # SRT/VTT generation
│   └── routers/
│       └── transcription.py               # UPDATED: Model docs
├── frontend/
│   └── src/
│       ├── pages/
│       │   └── SubtitleGeneratorPage.tsx  # NEW: Subtitle Generator UI
│       └── components/layout/
│           └── Layout.tsx                 # UPDATED: Added Subtitles tab
└── subtitle_gen.md                        # This file
```

---

## Installation

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .env/bin/activate  # Linux/macOS

# Install dependencies (already in requirements.txt)
pip install whisper soundfile scipy

# Or from requirements.txt
pip install -r requirements.txt
```

---

## Usage

### Via Frontend

1. Navigate to the **Subtitles** tab (🎬 icon)
2. Upload audio or video file
3. Select language (optional, auto-detect available)
4. Click "Generate Subtitles"
5. Preview in SRT or VTT format
6. Download or copy to clipboard

### Via API

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "model=parakeet-tdt-0.6b" \
  -F "language=en"
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="any-key"
)

with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        file=f,
        model="parakeet-tdt-0.6b",
        response_format="srt"
    )
    print(result)
```

---

## Subtitle Formats

### SRT (SubRip)
- Most widely supported format
- Compatible with all video players
- Format: `HH:MM:SS,mmm --> HH:MM:SS,mmm`

### VTT (WebVTT)
- Web standard for HTML5 video
- Supports additional styling
- Format: `HH:MM:SS.mmm --> HH:MM:SS.mmm`

---

## Future GPU Implementation (Parakeet TDT)

### Overview

For users with NVIDIA GPUs, implementing GPU acceleration via NVIDIA NeMo toolkit can provide:
- More accurate timestamps
- Better handling of long-form audio
- Faster inference (3-5x with GPU)

### Required Changes

#### 1. Update Dependencies

```txt
# GPU Implementation (future)
nemo-toolkit[asr]>=1.20.0
cuda-python  # For CUDA 12.x
```

#### 2. Update `parakeet_service.py`

```python
def load_parakeet_model_gpu():
    """Load Parakeet model using NVIDIA NeMo for GPU acceleration."""
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3"
    )
    return model
```

### Performance Comparison (Expected)

| Metric | Current (Whisper CPU) | Future (Parakeet GPU) |
|--------|----------------------|----------------------|
| Memory | ~1GB RAM | ~3GB VRAM |
| 1-min audio | ~20-40s | ~3-5s |
| 10-min audio | ~4-6 min | ~30-50s |
| Timestamp Accuracy | Good | Excellent |

---

## Troubleshooting

### Issue: Model download fails

**Solution**: Check internet connection. Whisper model is ~460MB.

### Issue: Out of memory

**Solution**: Close other applications. Reduce file size or split long audio.

### Issue: Slow inference

**Solution**: Normal for CPU. Use smaller model or consider GPU implementation.

### Issue: No segments in result

**Solution**: Ensure audio has speech content. Silent audio produces empty results.

---

## References

- [NVIDIA Parakeet TDT Model Card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/asr/models.html)
- [Parakeet-API GitHub](https://github.com/jianchang512/parakeet-api)
