#!/bin/bash
set -e

# Local development server script

MODEL_NAME=${1:-"clip"}
PORT=${2:-"8000"}
LOG_LEVEL=${3:-"INFO"}

echo "Starting local development server for: $MODEL_NAME"
echo "Port: $PORT"
echo "Log Level: $LOG_LEVEL"

# Check if model service exists
SERVICE_DIR="services/$MODEL_NAME"
if [ ! -d "$SERVICE_DIR" ]; then
    echo "Error: Model service '$MODEL_NAME' not found in $SERVICE_DIR"
    echo "Available services:"
    ls -1 services/ | grep -v __pycache__ | grep -v common
    exit 1
fi

# Check if app.py exists
APP_FILE="$SERVICE_DIR/app.py"
if [ ! -f "$APP_FILE" ]; then
    echo "Error: App file not found: $APP_FILE"
    exit 1
fi

# Set environment variables for local development
export MODEL_GCS_PATH="gs://model-zoo/dev/$MODEL_NAME/latest/"
export MODEL_NAME="$MODEL_NAME"
export DEVICE="cpu"  # Use CPU for local development
export HOST="0.0.0.0"
export PORT="$PORT"
export LOG_LEVEL="$LOG_LEVEL"
export CACHE_DIR="./cache"
export ENABLE_CACHE="true"

# Create cache directory
mkdir -p ./cache

echo ""
echo "Environment variables:"
echo "  MODEL_GCS_PATH: $MODEL_GCS_PATH"
echo "  MODEL_NAME: $MODEL_NAME"
echo "  DEVICE: $DEVICE"
echo "  PORT: $PORT"
echo "  LOG_LEVEL: $LOG_LEVEL"
echo ""

# Install dependencies if needed
if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found. Creating one..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
else
    source .venv/bin/activate
fi

# Run the service with hot reload
echo "Starting uvicorn server with hot reload..."
echo "API docs available at: http://localhost:$PORT/docs"
echo "Health check: http://localhost:$PORT/health"
echo ""

uvicorn "services.$MODEL_NAME.app:app" \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --log-level "$LOG_LEVEL"