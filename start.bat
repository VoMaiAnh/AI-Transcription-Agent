@echo off
REM Startup script for AI Transcription & TTS Agent

echo ============================================
echo AI Transcription & TTS Agent
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please create it with: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Start backend
echo.
echo Starting backend server...
echo API will be available at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo.
start "AI Transcription Backend" cmd /k "python -m app.main"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Check if frontend dependencies are installed
cd frontend
if not exist "node_modules" (
    echo.
    echo Installing frontend dependencies...
    call npm install
)

REM Start frontend
echo.
echo Starting frontend development server...
echo Frontend will be available at: http://localhost:3000
echo.
start "AI Transcription Frontend" cmd /k "npm run dev"

echo.
echo ============================================
echo Both servers are starting...
echo - Backend: http://localhost:8000
echo - Frontend: http://localhost:3000
echo ============================================
echo.
echo Press any key to exit this window...
pause >nul
