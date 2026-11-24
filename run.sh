#!/usr/bin/env bash
# ==============================================================================
# SIGNAL / VISION: Master Startup Script
# Starts both the FastAPI Backend (Port 8000) and the React Frontend (Port 5173)
# ==============================================================================

set -e

# Change directory to project root
cd "$(dirname "$0")"

echo "=================================================="
echo "Starting SIGNAL / VISION Platform"
echo "=================================================="

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found! Please create it first:"
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt"
    exit 1
fi

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI Backend
echo "[1/2] Launching FastAPI Backend on http://localhost:8000..."
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait briefly for backend to bind
sleep 2

# 2. Start Frontend Dev Server
echo "[2/2] Launching React Vite Frontend on http://localhost:5173..."
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo "SIGNAL / VISION is live!"
echo "• Frontend (HMR Development): http://localhost:5173"
echo "• Backend & Production SPA:   http://localhost:8000"
echo "• API Documentation (Swagger): http://localhost:8000/docs"
echo "=================================================="
echo "Press CTRL+C at any time to stop both servers."
echo ""

# Keep running
wait
