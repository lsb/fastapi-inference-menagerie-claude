#!/bin/bash
set -e

# Run all tests for the FastAPI Inference Menagerie

echo "🧪 Running all tests for FastAPI Inference Menagerie..."

# Set environment variables for testing
export DEVICE=cpu
export CACHE_DIR=/tmp/test_cache

# Create cache directory
mkdir -p "$CACHE_DIR"

echo ""
echo "📋 Test Environment:"
echo "  DEVICE: $DEVICE"
echo "  CACHE_DIR: $CACHE_DIR"
echo ""

# Run unit tests
echo "🔬 Running unit tests..."
echo "=================================="
pytest tests/unit/ -v

echo ""
echo "🔬 Running integration tests..."
echo "=================================="
if [ -d "tests/e2e" ]; then
    pytest tests/e2e/ -v
else
    echo "⏭️  No integration tests found (tests/e2e/ directory missing)"
fi

echo ""
echo "🔬 Running performance tests..."
echo "=================================="
if [ -d "tests/performance" ]; then
    pytest tests/performance/ -v
else
    echo "⏭️  No performance tests found (tests/performance/ directory missing)"
fi

echo ""
echo "✅ All tests completed!"

# Clean up
rm -rf "$CACHE_DIR"