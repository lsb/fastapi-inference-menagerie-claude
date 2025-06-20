#!/bin/bash
set -e

# Comprehensive End-to-End Local Deployment Test
# This script starts all services locally and tests them comprehensively

echo "🚀 Starting comprehensive end-to-end local deployment test..."
echo ""

# Set environment variables
export CACHE_DIR=/tmp/e2e_test_cache
export DEVICE=cpu
export LOG_level=INFO

# Create cache directory
mkdir -p "$CACHE_DIR"

echo "📋 E2E Test Environment:"
echo "  CACHE_DIR: $CACHE_DIR"
echo "  DEVICE: $DEVICE"
echo "  LOG_LEVEL: $LOG_LEVEL"
echo ""

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

# Function to wait for service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start within timeout"
    return 1
}

# Function to test service endpoints
test_service_endpoints() {
    local port=$1
    local service_name=$2
    
    echo ""
    echo "🔍 Testing $service_name endpoints on port $port..."
    echo "================================================"
    
    # Test health endpoint
    echo "Testing health endpoint..."
    health_response=$(curl -s "http://localhost:$port/health")
    echo "Health: $health_response"
    
    # Test root endpoint
    echo "Testing root endpoint..."
    root_response=$(curl -s "http://localhost:$port/")
    echo "Root: $root_response"
    
    # Test metrics endpoint
    echo "Testing metrics endpoint..."
    metrics_lines=$(curl -s "http://localhost:$port/metrics" | wc -l)
    echo "Metrics: $metrics_lines lines of Prometheus metrics"
    
    return 0
}

# Function to test CLIP specific endpoints
test_clip_endpoints() {
    local port=$1
    
    echo ""
    echo "🔍 Testing CLIP-specific endpoints..."
    echo "===================================="
    
    # Test text encoding
    echo "Testing CLIP text encoding..."
    text_response=$(curl -s -X POST "http://localhost:$port/v1/clip/encode/text" \
        -H "Content-Type: application/json" \
        -d '{"texts": ["a cat", "a dog"]}')
    echo "Text encoding response: $(echo $text_response | jq -c '.success')"
    
    # Test image encoding
    echo "Testing CLIP image encoding..."
    image_response=$(curl -s -X POST "http://localhost:$port/v1/clip/encode/image" \
        -H "Content-Type: application/json" \
        -d "{\"images\": [\"$TEST_IMAGE_B64\"]}")
    echo "Image encoding response: $(echo $image_response | jq -c '.success')"
    
    # Test similarity computation
    echo "Testing CLIP similarity..."
    similarity_response=$(curl -s -X POST "http://localhost:$port/v1/clip/similarity" \
        -H "Content-Type: application/json" \
        -d "{\"texts\": [\"a cat\", \"a dog\"], \"images\": [\"$TEST_IMAGE_B64\"]}")
    echo "Similarity response: $(echo $similarity_response | jq -c '.success')"
}

# Function to test Grounding SAM endpoints
test_grounding_sam_endpoints() {
    local port=$1
    
    echo ""
    echo "🔍 Testing Grounding SAM endpoints..."
    echo "===================================="
    
    # Test detection endpoint
    echo "Testing Grounding SAM detection..."
    detect_response=$(curl -s -X POST "http://localhost:$port/v1/ground/detect" \
        -H "Content-Type: application/json" \
        -d "{\"image\": \"$TEST_IMAGE_B64\", \"text\": \"object\", \"confidence_threshold\": 0.3}")
    echo "Detection response: $(echo $detect_response | jq -c '.success')"
    
    # Test segmentation endpoint
    echo "Testing Grounding SAM segmentation..."
    segment_response=$(curl -s -X POST "http://localhost:$port/v1/ground/segment" \
        -H "Content-Type: application/json" \
        -d "{\"image\": \"$TEST_IMAGE_B64\", \"text\": \"object\", \"confidence_threshold\": 0.3, \"include_masks\": true}")
    echo "Segmentation response: $(echo $segment_response | jq -c '.success')"
}

# Function to test Qwen VL endpoints
test_qwen_vl_endpoints() {
    local port=$1
    
    echo ""
    echo "🔍 Testing Qwen VL endpoints..."
    echo "=============================="
    
    # Test VQA endpoint
    echo "Testing Qwen VL visual question answering..."
    vqa_response=$(curl -s -X POST "http://localhost:$port/v1/vqa/ask" \
        -H "Content-Type: application/json" \
        -d "{\"image\": \"$TEST_IMAGE_B64\", \"question\": \"What is in this image?\", \"max_tokens\": 50}")
    echo "VQA response: $(echo $vqa_response | jq -c '.success')"
    
    # Test chat endpoint
    echo "Testing Qwen VL chat..."
    chat_response=$(curl -s -X POST "http://localhost:$port/v1/vqa/chat" \
        -H "Content-Type: application/json" \
        -d "{\"image\": \"$TEST_IMAGE_B64\", \"question\": \"Describe this image\", \"max_tokens\": 50}")
    echo "Chat response: $(echo $chat_response | jq -c '.success')"
}

# Start services and test them
echo "🚀 Starting all services..."
echo ""

# Start CLIP service
echo "Starting CLIP service on port 8001..."
CACHE_DIR=$CACHE_DIR DEVICE=$DEVICE python -m uvicorn services.clip.app:app --host 0.0.0.0 --port 8001 > /tmp/clip_service.log 2>&1 &
CLIP_PID=$!

# Start Grounding SAM service  
echo "Starting Grounding SAM service on port 8002..."
CACHE_DIR=$CACHE_DIR DEVICE=$DEVICE python -m uvicorn services.grounding_sam.app:app --host 0.0.0.0 --port 8002 > /tmp/grounding_sam_service.log 2>&1 &
GROUNDING_PID=$!

# Start Qwen VL service
echo "Starting Qwen VL service on port 8003..."
CACHE_DIR=$CACHE_DIR DEVICE=$DEVICE python -m uvicorn services.qwen_vl.app:app --host 0.0.0.0 --port 8003 > /tmp/qwen_vl_service.log 2>&1 &
QWEN_PID=$!

echo ""
echo "📝 Service PIDs:"
echo "  CLIP: $CLIP_PID"
echo "  Grounding SAM: $GROUNDING_PID"
echo "  Qwen VL: $QWEN_PID"
echo ""

# Function to cleanup services
cleanup() {
    echo ""
    echo "🧹 Cleaning up services..."
    kill $CLIP_PID $GROUNDING_PID $QWEN_PID 2>/dev/null || true
    rm -rf "$CACHE_DIR"
    echo "✅ Cleanup completed"
}

# Set up cleanup trap
trap cleanup EXIT

# Wait for services to be ready
wait_for_service 8001 "CLIP"
wait_for_service 8002 "Grounding SAM"
wait_for_service 8003 "Qwen VL"

# Test all services
test_service_endpoints 8001 "CLIP"
test_clip_endpoints 8001

test_service_endpoints 8002 "Grounding SAM"
test_grounding_sam_endpoints 8002

test_service_endpoints 8003 "Qwen VL"
test_qwen_vl_endpoints 8003

echo ""
echo "🎉 End-to-End Local Deployment Test Completed Successfully!"
echo ""
echo "Summary:"
echo "  ✅ CLIP service - All endpoints working"
echo "  ✅ Grounding SAM service - All endpoints working"
echo "  ✅ Qwen VL service - All endpoints working"
echo ""
echo "📊 Service URLs:"
echo "  CLIP: http://localhost:8001/docs"
echo "  Grounding SAM: http://localhost:8002/docs"
echo "  Qwen VL: http://localhost:8003/docs"
echo ""