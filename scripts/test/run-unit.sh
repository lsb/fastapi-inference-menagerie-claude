#!/bin/bash
set -e

# Run unit tests for specific models

echo "🔬 Running unit tests for FastAPI Inference Menagerie..."

# Set environment variables for testing
export DEVICE=cpu
export CACHE_DIR=/tmp/test_cache

# Create cache directory
mkdir -p "$CACHE_DIR"

# Parse arguments
MODEL_FILTER=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--model MODEL] [-v|--verbose]"
            echo ""
            echo "Options:"
            echo "  --model MODEL    Run tests for specific model (clip, grounding_sam, qwen_vl)"
            echo "  -v, --verbose    Verbose output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all unit tests"
            echo "  $0 --model clip              # Run only CLIP tests"
            echo "  $0 --model grounding_sam -v  # Run Grounding SAM tests with verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "📋 Test Environment:"
echo "  DEVICE: $DEVICE"
echo "  CACHE_DIR: $CACHE_DIR"
if [ -n "$MODEL_FILTER" ]; then
    echo "  MODEL FILTER: $MODEL_FILTER"
fi
echo ""

# Determine test path
if [ -n "$MODEL_FILTER" ]; then
    case $MODEL_FILTER in
        clip)
            TEST_PATH="tests/unit/test_image_classification_simple.py"
            ;;
        grounding_sam)
            TEST_PATH="tests/unit/test_grounding_sam_service.py tests/unit/test_grounding_sam_adapter.py"
            ;;
        qwen_vl)
            TEST_PATH="tests/unit/test_qwen_vl_service.py tests/unit/test_qwen_vl_adapter.py"
            ;;
        *)
            echo "❌ Unknown model: $MODEL_FILTER"
            echo "Available models: clip, grounding_sam, qwen_vl"
            exit 1
            ;;
    esac
else
    TEST_PATH="tests/unit/"
fi

# Run the tests
echo "🔬 Running unit tests..."
echo "=================================="
pytest $TEST_PATH $VERBOSE

echo ""
echo "✅ Unit tests completed!"

# Clean up
rm -rf "$CACHE_DIR"