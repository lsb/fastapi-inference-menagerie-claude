#!/bin/bash
set -e

# Run tests in CPU mode to verify all models work without GPU

echo "💻 Running tests in CPU mode..."
echo ""
echo "This script verifies that all models can run in CPU-only environments"
echo "without requiring GPU hardware or CUDA."
echo ""

# Set environment variables for CPU testing
export DEVICE=cpu
export CACHE_DIR=/tmp/test_cache

# Create cache directory
mkdir -p "$CACHE_DIR"

echo "📋 CPU Test Environment:"
echo "  DEVICE: $DEVICE"
echo "  CACHE_DIR: $CACHE_DIR"
echo ""

# Test each model separately
echo "🔬 Testing CLIP model in CPU mode..."
echo "=================================="
pytest tests/unit/test_image_classification_simple.py -v

echo ""
echo "🔬 Testing Grounding DINO + SAM2 model in CPU mode..."
echo "=================================="
pytest tests/unit/test_grounding_sam_service.py tests/unit/test_grounding_sam_adapter.py -v

echo ""
echo "🔬 Testing Qwen VL model in CPU mode..."
echo "=================================="
pytest tests/unit/test_qwen_vl_service.py tests/unit/test_qwen_vl_adapter.py -v

echo ""
echo "✅ All models work correctly in CPU mode!"
echo ""
echo "Summary:"
echo "  ✅ CLIP - CPU compatible"
echo "  ✅ Grounding DINO + SAM2 - CPU compatible"  
echo "  ✅ Qwen VL - CPU compatible"
echo ""
echo "🎉 All models can be deployed in CPU-only environments!"

# Clean up
rm -rf "$CACHE_DIR"