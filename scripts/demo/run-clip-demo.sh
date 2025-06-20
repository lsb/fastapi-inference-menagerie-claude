#!/bin/bash
set -e

# Demo script to run CLIP service locally and test it

echo "🦁 Running CLIP Service Demo..."
echo ""
echo "This script will:"
echo "  1. Start the CLIP service locally"
echo "  2. Wait for it to be ready"
echo "  3. Test the API endpoints"
echo "  4. Shut down the service"
echo ""

# Set environment variables
export DEVICE=cpu
export CACHE_DIR=./cache
export MODEL_NAME=clip
export PORT=8000

# Create cache directory
mkdir -p $CACHE_DIR

# Start the service in background
echo "🚀 Starting CLIP service on port $PORT..."
uvicorn services.clip.app:app --host 0.0.0.0 --port $PORT --log-level ERROR &
SERVICE_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    kill $SERVICE_PID 2>/dev/null || true
    wait $SERVICE_PID 2>/dev/null || true
    echo "✅ Service stopped"
}
trap cleanup EXIT

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "✅ Service is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Service failed to start"
        exit 1
    fi
    sleep 1
done

# Test the API
echo ""
echo "🧪 Testing API endpoints..."
echo ""

# Test health check
echo "1️⃣ Health Check:"
curl -s http://localhost:$PORT/health | jq '.'

# Test root endpoint
echo ""
echo "2️⃣ Root Endpoint:"
curl -s http://localhost:$PORT/ | jq '.'

# Test text encoding
echo ""
echo "3️⃣ Text Encoding (cat vs dog):"
curl -s -X POST http://localhost:$PORT/v1/clip/encode/text \
    -H "Content-Type: application/json" \
    -d '{"texts": ["a photo of a cat", "a photo of a dog"]}' | jq '.'

# Test docs
echo ""
echo "4️⃣ API Documentation available at: http://localhost:$PORT/docs"
echo ""

echo "✅ Demo completed successfully!"
echo ""
echo "The service is still running. Press Ctrl+C to stop it."
echo ""

# Keep the script running
wait $SERVICE_PID