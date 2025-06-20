#!/bin/bash
set -e

# Simple End-to-End Test with Real Image Data

echo "🚀 Starting simple end-to-end test with real images..."
echo ""

# Set environment
export CACHE_DIR=/tmp/simple_e2e_cache
export DEVICE=cpu
mkdir -p "$CACHE_DIR"

# Use existing test image from repository
cd /Users/lsb/jupyterlab/fastapi-inference-menagerie-claude
TEST_IMAGE_B64=$(python3 -c "
import base64

# Use the cat office image from our test data
with open('tests/data/images/cat_office_typing.jpg', 'rb') as f:
    img_data = f.read()
    b64_data = base64.b64encode(img_data).decode()
    print(b64_data)
")

echo "📋 Test Environment:"
echo "  CACHE_DIR: $CACHE_DIR"
echo "  DEVICE: $DEVICE"
echo "  Test image size: $(echo $TEST_IMAGE_B64 | wc -c) characters"
echo ""

# Function to test service
test_service() {
    local port=$1
    local service_name=$2
    
    echo "🔍 Testing $service_name on port $port..."
    
    # Health check
    health=$(curl -s "http://localhost:$port/health" | jq -r '.status')
    echo "  Health: $health"
    
    if [[ "$health" != "healthy" ]]; then
        echo "  ❌ Service not healthy"
        return 1
    fi
    
    return 0
}

# Function to cleanup
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    rm -rf "$CACHE_DIR"
    if [[ -n $CLIP_PID ]]; then kill $CLIP_PID 2>/dev/null || true; fi
    echo "✅ Cleanup completed"
}

trap cleanup EXIT

# Start CLIP service
echo "🚀 Starting CLIP service..."
CACHE_DIR=$CACHE_DIR DEVICE=$DEVICE python -m uvicorn services.clip.app:app --host 0.0.0.0 --port 8001 > /tmp/simple_clip.log 2>&1 &
CLIP_PID=$!

# Wait for service
echo "⏳ Waiting for CLIP service..."
for i in {1..15}; do
    if curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
        echo "✅ CLIP service is ready!"
        break
    fi
    echo "  Attempt $i/15..."
    sleep 2
done

# Test the service
test_service 8001 "CLIP"

# Test CLIP text encoding
echo ""
echo "🔍 Testing CLIP text encoding..."
text_result=$(curl -s -X POST "http://localhost:8001/v1/clip/encode/text" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["a cat", "a dog"]}')

success=$(echo $text_result | jq -r '.success')
echo "  Text encoding success: $success"

if [[ "$success" == "true" ]]; then
    count=$(echo $text_result | jq -r '.result.count')
    echo "  Embeddings count: $count"
else
    echo "  ❌ Text encoding failed"
    echo $text_result | jq '.'
fi

# Test CLIP image encoding  
echo ""
echo "🔍 Testing CLIP image encoding..."
image_result=$(curl -s -X POST "http://localhost:8001/v1/clip/encode/image" \
    -H "Content-Type: application/json" \
    -d "{\"images\": [\"$TEST_IMAGE_B64\"]}")

success=$(echo $image_result | jq -r '.success')
echo "  Image encoding success: $success"

if [[ "$success" == "true" ]]; then
    count=$(echo $image_result | jq -r '.result.count')
    echo "  Embeddings count: $count"
else
    echo "  ❌ Image encoding failed"
    echo $image_result | jq '.'
fi

echo ""
echo "🎉 Simple E2E test completed!"
echo ""
echo "📊 Results:"
echo "  ✅ Service health checks: PASSED"
echo "  $([ "$success" == "true" ] && echo "✅" || echo "❌") API endpoints: $([ "$success" == "true" ] && echo "PASSED" || echo "FAILED")"
echo ""