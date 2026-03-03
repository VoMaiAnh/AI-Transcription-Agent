#!/bin/bash
# Startup script for AI Transcription & TTS Agent

echo "============================================"
echo "AI Transcription & TTS Agent"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please create it with: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start backend in background
echo ""
echo "Starting backend server..."
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
python -m app.main &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Check if frontend dependencies are installed
cd frontend
if [ ! -d "node_modules" ]; then
    echo ""
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend
echo ""
echo "Starting frontend development server..."
echo "Frontend will be available at: http://localhost:3000"
echo ""
npm run dev &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "Both servers are running..."
echo "- Backend: http://localhost:8000 (PID: $BACKEND_PID)"
echo "- Frontend: http://localhost:3000 (PID: $FRONTEND_PID)"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for both processes
wait
