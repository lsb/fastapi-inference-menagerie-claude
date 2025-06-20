#!/bin/bash

# Environment setup script for development

echo "🔧 Setting up development environment variables..."

# Default configuration
export MODEL_GCS_PATH="gs://model-zoo/dev/models/"
export MODEL_NAME="clip"
export DEVICE="cpu"
export LOG_LEVEL="INFO"
export CACHE_DIR="./cache"
export ENABLE_CACHE="true"
export HOST="0.0.0.0"
export PORT="8000"

# Create cache directory
mkdir -p "$CACHE_DIR"

echo ""
echo "📋 Environment Variables:"
echo "  MODEL_GCS_PATH: $MODEL_GCS_PATH"
echo "  MODEL_NAME: $MODEL_NAME"
echo "  DEVICE: $DEVICE"
echo "  LOG_LEVEL: $LOG_LEVEL"
echo "  CACHE_DIR: $CACHE_DIR"
echo "  ENABLE_CACHE: $ENABLE_CACHE"
echo "  HOST: $HOST"
echo "  PORT: $PORT"
echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To use these variables in your shell, run:"
echo "  source scripts/setup/environment.sh"