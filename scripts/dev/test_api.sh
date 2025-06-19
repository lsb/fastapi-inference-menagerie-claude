#!/bin/bash
set -e

# API testing script for local development

SERVICE_URL=${1:-"http://localhost:8000"}
MODEL_NAME=${2:-"clip"}

echo "Testing API at: $SERVICE_URL"
echo "Model: $MODEL_NAME"

# Test health endpoint
echo ""
echo "🔍 Testing health endpoint..."
curl -s "$SERVICE_URL/health" | jq '.' || echo "Health check failed"

# Test root endpoint
echo ""
echo "🔍 Testing root endpoint..."
curl -s "$SERVICE_URL/" | jq '.' || echo "Root endpoint failed"

# Test metrics endpoint
echo ""
echo "🔍 Testing metrics endpoint..."
curl -s "$SERVICE_URL/metrics" | head -10 || echo "Metrics endpoint failed"

# Model-specific tests
case $MODEL_NAME in
    "clip")
        echo ""
        echo "🔍 Testing CLIP text encoding..."
        curl -s -X POST "$SERVICE_URL/v1/clip/encode/text" \
            -H "Content-Type: application/json" \
            -d '{"texts": ["a cat", "a dog"]}' | jq '.' || echo "CLIP text encoding failed"
        ;;
    
    "grounding-sam" | "grounding_sam")
        echo ""
        echo "🔍 Testing Grounding SAM detection..."
        # Note: This would need a base64 encoded image
        echo "Grounding SAM test requires base64 image - skipping"
        ;;
    
    "qwen-vl" | "qwen_vl")
        echo ""
        echo "🔍 Testing Qwen VL chat..."
        # Note: This would need a base64 encoded image
        echo "Qwen VL test requires base64 image - skipping"
        ;;
    
    *)
        echo ""
        echo "🔍 Testing generic predict endpoint..."
        curl -s -X POST "$SERVICE_URL/v1/$MODEL_NAME/predict" \
            -H "Content-Type: application/json" \
            -d '{"input_data": {"test": "data"}}' | jq '.' || echo "Generic predict failed"
        ;;
esac

echo ""
echo "✅ API testing completed!"
echo ""
echo "For interactive testing, visit: $SERVICE_URL/docs"