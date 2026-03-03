# AI Transcription & TTS Agent

An AI-powered audio/video transcription and text-to-speech application using Whisper, Qwen3-ASR, and Qwen3-TTS models. This tool converts audio and video files into accurate text transcriptions and can also synthesize speech from text.
<img width="1908" height="906" alt="Captura de pantalla 2026-03-03 232938" src="https://github.com/user-attachments/assets/f9f842a5-097c-40cc-95ad-856908daa4c9" />

## Features

### Speech-to-Text (STT)
- **High-accuracy speech-to-text transcription** using OpenAI Whisper
- **Qwen3-ASR models** with support for 30+ languages and 22 Chinese dialects
- **Multiple model options** from fast (tiny) to high-accuracy (large)
- **Video file support** with automatic audio extraction
- **Subtitle generation** in SRT and VTT formats

### Text-to-Speech (TTS)
- **Qwen3-TTS models** for natural-sounding speech synthesis
- **CosyVoice models** for high-quality voice generation
- **Voice selection** with multiple options
- **Speed and pitch control** for customized output

### Web Interface
- **Modern React frontend** with Vite for fast development
- **Responsive design** that works on all devices
- **Real-time progress** indicators
- **History management** for transcriptions and TTS results
  
<p align="center" style="display: flex; flex-direction: row; overflow-x: auto; white-space: nowrap;">
  <img src="https://github.com/user-attachments/assets/939e94cf-5a22-4fdb-822b-59ae56518074" width="600" alt="Transcription-tab" />
  <img src="https://github.com/user-attachments/assets/15d54856-d612-4ca8-8a6d-44eee56ff993" width="600" alt="TTS-tab" />
  <img src="https://github.com/user-attachments/assets/c8ba049a-555f-44ce-ba88-92c15eba9d2e" width="600" alt="History-tab" />
</p>

## Project Structure

```
AI-Transcription-Agent/
├── app/                    # Backend (FastAPI)
│   ├── __init__.py
│   ├── main.py            # FastAPI application entry
│   ├── config.py          # Configuration and settings
│   ├── models/            # Pydantic models
│   │   ├── transcription.py
│   │   └── tts.py
│   ├── routers/           # API routers
│   │   ├── __init__.py
│   │   ├── transcription.py
│   │   └── tts.py
│   ├── services/          # Business logic
│   │   ├── __init__.py
│   │   ├── transcription_service.py
│   │   ├── tts_service.py
│   │   └── subtitle_service.py
│   ├── utils/             # Utilities
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   └── file_utils.py
│   └── storage/           # File storage
│       ├── __init__.py
│       └── local.py
├── frontend/              # Frontend (React + Vite)
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       ├── api/
│       └── types/
├── uploads/               # Uploaded files storage
├── requirements.txt       # Python dependencies
└── README.md
```

## Prerequisites

Before running this project, ensure you have:

- **Python 3.10+** (including Python 3.12+)
- **FFmpeg** (Required for audio processing)
- **Git** (For cloning the repository)
- **Node.js 18+** (For frontend development)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VoMaiAnh/AI-Transcription-Agent.git
cd AI-Transcription-Agent
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

**Activate the virtual environment:**

- Windows:
  ```bash
  .venv\Scripts\activate
  ```
- Linux/macOS:
  ```bash
  source .venv/bin/activate
  ```

### 3. Upgrade pip and setuptools

```bash
pip install --upgrade pip setuptools wheel
```

> This prevents `ModuleNotFoundError: No module named 'pkg_resources'` errors during installation.

### 4. Install OpenAI Whisper

Install Whisper directly from GitHub for best compatibility:

```bash
pip install git+https://github.com/openai/whisper.git
```

### 5. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 6. Install FFmpeg

**Windows (using winget):**
```bash
winget install --id=Gyan.FFmpeg -e
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 7. Verify FFmpeg Installation

```bash
ffmpeg -version
```

If the command doesn't work, restart your terminal and try again.

### 8. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# STT Model Configuration
STT_MODEL=base
DEVICE=cpu
WHISPER_MODEL=base
QWEN3_ASR_MODEL=Qwen/Qwen3-ASR-1.7B

# TTS Model Configuration
TTS_MODEL=qwen3-tts-1.8b

# Application settings
MAX_FILE_SIZE=52428800
UPLOAD_DIR=./uploads

# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### FFmpeg Path Configuration (Windows Only)

If you encounter the warning `Couldn't find ffmpeg or avconv`, you need to configure the FFmpeg path:

**Find your FFmpeg installation path:**
```bash
where ffmpeg
```

**Add to your Python script (if needed):**
```python
from pydub import AudioSegment

# Replace with your actual FFmpeg path
AudioSegment.converter = r"C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-x.x.x-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-x.x.x-full_build\bin\ffprobe.exe"
```

## Usage

### Running the Backend

```bash
# From the project root
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Running the Frontend

```bash
# From the frontend directory
cd frontend
npm run dev
```

The frontend will be available at http://localhost:3000

### Building the Frontend for Production

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`.

### Using the API Directly

#### Transcribe Audio/Video

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "model=whisper-base"
```

#### Synthesize Speech

```bash
curl -X POST http://localhost:8000/api/v1/tts/synthesize \
  -F "text=Hello, this is a test." \
  -F "model=qwen3-tts-1.8b" \
  -o output.wav
```

#### Download Subtitle

```bash
curl -X POST http://localhost:8000/api/v1/subtitle/{transcription_id} \
  -F "format=srt" \
  -o subtitle.srt
```

## API Endpoints

### Transcription

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models` | List available STT models |
| POST | `/api/v1/transcribe` | Transcribe audio/video file |
| GET | `/api/v1/transcription/{id}` | Get transcription by ID |
| DELETE | `/api/v1/transcription/{id}` | Delete transcription |
| GET | `/api/v1/list` | List all transcriptions |
| POST | `/api/v1/subtitle/{id}` | Download subtitle (SRT/VTT) |

### Text-to-Speech

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tts/models` | List TTS models |
| GET | `/api/v1/tts/voices` | List available voices |
| POST | `/api/v1/tts/synthesize` | Synthesize speech from text |
| GET | `/api/v1/tts/result/{id}` | Get TTS result by ID |
| DELETE | `/api/v1/tts/result/{id}` | Delete TTS result |
| GET | `/api/v1/tts/list` | List all TTS results |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |

## Supported Formats

### Audio Formats
- MP3, WAV, FLAC, OGG, M4A, AAC

### Video Formats
- MP4, MOV, MKV, WEBM, AVI

### Subtitle Formats
- SRT (SubRip)
- VTT (WebVTT)

### Maximum File Size
- Default: 50 MB

## System Requirements

- **OS:** Windows 10/11, macOS 10.15+, or Linux
- **RAM:** Minimum 8GB (16GB recommended for larger models)
- **Storage:** At least 5GB free space for models and dependencies
- **GPU:** CUDA-compatible GPU recommended for faster processing (optional)

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pkg_resources'`

**Solution:**
```bash
pip install --upgrade setuptools wheel
pip install git+https://github.com/openai/whisper.git
```

### Issue: `pydub - Couldn't find ffmpeg or avconv`

**Solution:**
1. Verify FFmpeg is installed: `ffmpeg -version`
2. Find FFmpeg path: `where ffmpeg` (Windows) or `which ffmpeg` (Linux/macOS)
3. Configure the path in your code (see Configuration section)
4. Or add FFmpeg to your system PATH environment variable

### Issue: Installing from requirements.txt fails

**Solution:**
If installing `openai-whisper` from requirements.txt fails with a build error, install it directly from GitHub:
```bash
pip install git+https://github.com/openai/whisper.git
pip install -r requirements.txt
```

### Issue: Can't find FFmpeg installation path

**Solution:**
Open Command Prompt and run:
```bash
where ffmpeg
```

If nothing appears, navigate to:
```
%LOCALAPPDATA%\Microsoft\WinGet\Packages
```
Look for the folder starting with `Gyan.FFmpeg` and navigate to the `bin` subfolder inside.

## Key Dependencies

- **openai-whisper** - OpenAI's Whisper model for speech recognition
- **pydub** - Audio manipulation and processing
- **pydantic-settings** - Configuration management
- **FFmpeg** - Backend for audio/video processing
- **React + Vite** - Modern frontend framework

See `requirements.txt` for the complete list of Python dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here - e.g., MIT, Apache 2.0]

## Author

**VoMaiAnh**
- GitHub: [@VoMaiAnh](https://github.com/VoMaiAnh)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the powerful transcription model
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) for multilingual ASR [github](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for text-to-speech [github](https://github.com/QwenLM/Qwen3-TTS)
- [pydub](https://github.com/jiaaro/pydub) for audio processing capabilities
- [FFmpeg](https://ffmpeg.org/) for multimedia framework

## Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open an [issue](https://github.com/VoMaiAnh/AI-Transcription-Agent/issues) on GitHub [github](https://github.com/QwenLM/Qwen3-ASR)
