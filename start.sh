#!/bin/bash

echo "🚀 Starting Face Detection Web App..."
echo "======================================"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    echo "🔄 Checking if port $port is in use..."
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "⚠️  Killing existing process on port $port..."
        lsof -ti:$port | xargs kill -9 >/dev/null 2>&1
        sleep 2
    fi
}

# Check if Python 3 is installed
if ! command_exists python3; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if Node.js is installed
if ! command_exists npm; then
    echo "❌ Error: Node.js/npm is not installed. Please install Node.js first."
    exit 1
fi

# Kill any existing processes on our ports
kill_port 5050
kill_port 3000
kill_port 3001

echo "📦 Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Always use venv's python and pip explicitly to avoid PATH issues
VENV_PY="$(pwd)/venv/bin/python"
VENV_PIP="$(pwd)/venv/bin/pip"

# Verify we are using correct interpreters
echo "🔧 Using Python: $VENV_PY"
$VENV_PY --version

# Install Python dependencies
echo "📚 Installing Python dependencies..."
$VENV_PIP install --quiet -r requirements.txt

# Install enhanced AI features
echo "🧠 Installing AI analysis features..."
$VENV_PIP install --quiet deepface tensorflow opencv-python tf-keras

# Test if DeepFace is working (one-liner to keep exit codes simple)
echo "🔬 Testing DeepFace installation..."
$VENV_PY -c "import sys, traceback;\ntry:\n from deepface import DeepFace; print('✅ DeepFace test passed!');\nexcept Exception as e:\n traceback.print_exc(); sys.exit(1)" && echo "✅ DeepFace is working correctly!" || { echo "❌ DeepFace test failed – reinstalling..."; $VENV_PIP install --force-reinstall deepface; }

echo "📱 Setting up Node.js dependencies..."

# Install Node.js dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install --silent
fi

# Ensure lucide-react is installed
npm install --silent lucide-react

echo "🎯 Starting backend server..."

# Start backend in background
nohup $VENV_PY app.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "❌ Backend failed to start. Check backend.log for details."
    cat backend.log
    exit 1
fi

echo "✅ Backend started successfully on port 5050"

echo "🎨 Starting frontend server..."

# Start frontend in background
nohup npm run start > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

echo "✅ Frontend started successfully"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    kill_port 5050
    kill_port 3000
    kill_port 3001
    echo "👋 Goodbye!"
}

# Set up cleanup on script exit
trap cleanup EXIT INT TERM

echo ""
echo "🎉 Face Detection Web App is now running!"
echo "======================================"
echo "🌐 Frontend: http://localhost:3000 (or http://localhost:3001)"
echo "🔧 Backend:  http://localhost:5050"
echo ""
echo "📊 Features available:"
echo "   ✅ Face Detection"
echo "   ✅ Emotion Analysis" 
echo "   ✅ Age Estimation"
echo "   ✅ Gender Classification"
echo "   ✅ Autonomous Learning System"
echo "   ✅ Modern React UI with animations"
echo ""
echo "⏳ Note: AI models are loading in the background..."
echo "   🔍 Check readiness: curl http://localhost:5050/ready"
echo "   ⚠️  Wait for 'All models ready' message before uploading images"
echo "   📊 This usually takes 30-60 seconds for first-time setup"
echo ""
echo "📝 Logs:"
echo "   Backend: tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Keep script running and monitor processes
while true; do
    if ! ps -p $BACKEND_PID > /dev/null; then
        echo "❌ Backend crashed! Check backend.log"
        break
    fi
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo "❌ Frontend crashed! Check frontend.log"
        break
    fi
    sleep 5
done 