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
    if echo "$response" | grep -q '"ready": true'; then
        echo "🎉 All AI models are loaded and ready!"
        echo "🚀 You can now upload images for face detection"
    elif echo "$response" | grep -q '"ready": false'; then
        echo "⏳ Models are still loading..."
        echo "   Please wait a moment and try again"
        echo "   Status: $(echo "$response" | grep -o '"status": "[^"]*"' | cut -d'"' -f4)"
    else
        echo "⚠️  Unexpected response from server:"
        echo "$response"
    fi
else
    echo "❌ Failed to check model readiness"
    echo "   Make sure the backend server is running"
fi 