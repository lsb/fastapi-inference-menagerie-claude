#!/bin/bash
set -e

# Run service tests in CPU mode to verify API endpoints work without GPU

echo "💻 Testing service endpoints in CPU mode..."
echo ""
echo "This script verifies that all model service APIs work correctly"
echo "in CPU-only environments without requiring GPU hardware."
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

# Test each model service separately
echo "🔬 Testing CLIP service endpoints..."
echo "=================================="
pytest tests/unit/test_image_classification_simple.py -v

echo ""
echo "🔬 Testing Grounding DINO + SAM2 service endpoints..."
echo "=================================="
pytest tests/unit/test_grounding_sam_service.py -v

echo ""
echo "🔬 Testing Qwen VL service endpoints..."
echo "=================================="
pytest tests/unit/test_qwen_vl_service.py -v

echo ""
echo "✅ All service endpoints work correctly in CPU mode!"
echo ""
echo "Summary:"
echo "  ✅ CLIP service - CPU compatible"
echo "  ✅ Grounding DINO + SAM2 service - CPU compatible"  
echo "  ✅ Qwen VL service - CPU compatible"
echo ""
echo "🎉 All model services can be deployed in CPU-only environments!"

# Clean up
rm -rf "$CACHE_DIR"