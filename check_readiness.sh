#!/bin/bash

echo "🔍 Checking AI model readiness..."

# Check if backend is running
if ! curl -s http://localhost:5050/health > /dev/null; then
    echo "❌ Backend server is not running on port 5050"
    echo "   Start the server with: ./start.sh"
    exit 1
fi

# Check model readiness
response=$(curl -s http://localhost:5050/ready)

if [ $? -eq 0 ]; then
    echo "✅ Backend server is responding"
    
    # Parse JSON response (basic check)
    status=$(echo "$response" | sed -nE 's/.*"status"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p')
    if echo "$response" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' && [ "$status" = "ready" ]; then
        echo "🎉 All AI models are loaded and ready!"
        echo "🚀 You can now upload images for face detection"
    elif echo "$response" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' && [ "$status" = "degraded" ]; then
        echo "✅ OpenCV detection is ready"
        echo "⚠️  DeepFace analysis is unavailable; continuing without attributes"
    elif echo "$response" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*false'; then
        case "$status" in
            loading)
                echo "⏳ Models are still loading..."
                echo "   Please wait a moment and try again"
                ;;
            error)
                echo "❌ Model initialization failed"
                echo "   Check backend.log, then restart the backend"
                ;;
            *)
                echo "⚠️  Models are not ready (status: $status)"
                ;;
        esac
    else
        echo "⚠️  Unexpected response from server:"
        echo "$response"
    fi
else
    echo "❌ Failed to check model readiness"
    echo "   Make sure the backend server is running"
fi 