# SYNCRA - AI Media Studio 

<img width="1672" height="941" alt="Logo2_long" src="https://github.com/user-attachments/assets/c04827c7-e05f-445b-861c-cc97a81400b1" />

An AI-powered audio/video transcription and text-to-speech application using Whisper, Parakeet TDT, and Supertonic 3. This tool converts audio and video files into accurate text transcriptions and can synthesize on-device preset voices from text.
<img width="1908" height="906" alt="Captura de pantalla 2026-03-03 232938" src="https://github.com/user-attachments/assets/f9f842a5-097c-40cc-95ad-856908daa4c9" />

## Features

### Speech-to-Text (STT)
- **High-accuracy speech-to-text transcription** using OpenAI Whisper
- **Parakeet TDT** for precise timestamped subtitle workflows
- **Multiple model options** from fast (tiny) to high-accuracy (large)
- **Video file support** with automatic audio extraction
- **Subtitle generation** in SRT and VTT formats

### Text-to-Speech (TTS)
- **Supertonic 3** for ONNX Runtime on-device multilingual speech synthesis
- **Built-in voice styles** M1-M5 and F1-F5 across 31 supported language codes
- **Speed control** for customized output

### Web Interface
- **Modern React frontend** with Vite for fast development
- **Responsive design** that works on all devices
- **Real-time progress** indicators
- **History management** for transcriptions and TTS results

### 🖥️ App Screenshots

<details>
  <summary>View Project Dashboard</summary>
  <img width="955" height="468" alt="Captura de pantalla 2026-06-07 104000" src="https://github.com/user-attachments/assets/db53367a-fc9b-4ce8-9d63-b68f169691b4" />
</details>

<details>
  <summary>View Live Editor Tab</summary>
  <img width="951" height="467" alt="Captura de pantalla 2026-06-07 104319" src="https://github.com/user-attachments/assets/1f697836-433b-4f11-b5ed-8de4535e51ea" />
</details>

<details>
  <summary>View Dubbing Studio Tab</summary>
  <img width="947" height="463" alt="Captura de pantalla 2026-06-07 103856" src="https://github.com/user-attachments/assets/dc884a54-42ca-4fb0-ac8d-7c6840d56357" />
  <img width="949" height="464" alt="Captura de pantalla 2026-06-07 103943" src="https://github.com/user-attachments/assets/b54e45de-d21d-47c9-8eec-c926d6863b8d" />
</details>

<details>
  <summary>View Archive & History Tab</summary>
  <img width="952" height="471" alt="Captura de pantalla 2026-06-07 104336" src="https://github.com/user-attachments/assets/409cc46e-672e-4d6a-82dc-9f0555e98a29" />
</details>

<details>
  <summary>View AI Tools Status Tab</summary>
  <img width="953" height="467" alt="Captura de pantalla 2026-06-07 104347" src="https://github.com/user-attachments/assets/c07ee7a3-e227-492a-b721-6877b86a5617" />
</details>

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

- **Python 3.10+**
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

### 4. Optional: Install OpenAI Whisper Directly

`requirements.txt` installs `openai-whisper`. If your platform needs the GitHub build, install it directly:

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
PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3

# TTS Model Configuration
TTS_MODEL=Supertone/supertonic-3

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
  -F "model=Supertone/supertonic-3" \
  -F "voice=M1" \
  -F "language=en" \
  -F "speed=1.0" \
  -o output.wav
```

#### Download Subtitle

```bash
curl "http://localhost:8000/api/v1/subtitle/{transcription_id}?format=srt" \
  -o subtitle.srt
```

#### Create Subtitled Video

```bash
curl -X POST http://localhost:8000/api/v1/subtitle/{transcription_id}/embed \
  -F "mode=soft" \
  -F "format=srt" \
  -o subtitled-video.mkv
```

Use `mode=hard` to burn subtitles into an MP4.

#### Create Dubbed Video

```bash
curl -X POST http://localhost:8000/api/v1/dub/{transcription_id} \
  -F "target_language=en" \
  -F "voice=default" \
  -o dubbed-video.mp4
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
| GET | `/api/v1/subtitle/{id}?format=srt` | Download subtitle (SRT/VTT) |
| POST | `/api/v1/subtitle/{id}` | Backward-compatible subtitle download |
| POST | `/api/v1/subtitle/{id}/embed` | Create soft or hard subtitled video |
| POST | `/api/v1/dub/{id}` | Create first-pass dubbed video |

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
- **supertonic** - ONNX Runtime on-device multilingual text-to-speech
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
- [Supertonic 3](https://github.com/supertone-inc/supertonic) for on-device multilingual TTS
- [pydub](https://github.com/jiaaro/pydub) for audio processing capabilities
- [FFmpeg](https://ffmpeg.org/) for multimedia framework

## Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open an [issue](https://github.com/VoMaiAnh/AI-Transcription-Agent/issues) on GitHub
