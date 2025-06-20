#!/usr/bin/env python3
"""Test all service endpoints with mock data."""

import base64
from io import BytesIO
from PIL import Image
import json

def create_test_image():
    """Create a simple test image and return as base64."""
    img = Image.new('RGB', (100, 100), color='blue')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode()

def test_clip_endpoints():
    """Test CLIP endpoints with mock client."""
    print("\n🎨 Testing CLIP Service...")
    print("-" * 40)
    
    # Mock text encoding request
    text_request = {
        "texts": ["a photo of a cat", "a photo of a dog", "a blue square"]
    }
    print("📝 Text Encoding Request:")
    print(f"   Texts: {text_request['texts']}")
    print("   Expected: 3 embeddings of dimension 512")
    
    # Mock image encoding request
    print("\n🖼️ Image Encoding Request:")
    print("   Image: base64 encoded 100x100 blue square")
    print("   Expected: 1 embedding of dimension 512")
    
    # Mock similarity request
    print("\n🔍 Similarity Request:")
    print("   Texts: ['a blue square', 'a red circle']")
    print("   Images: 1 blue square image")
    print("   Expected: 2x1 similarity matrix")

def test_grounding_sam_endpoints():
    """Test Grounding DINO + SAM2 endpoints."""
    print("\n🎯 Testing Grounding DINO + SAM2 Service...")
    print("-" * 40)
    
    # Mock detection request
    detect_request = {
        "image": "base64_image_data",
        "text": "person walking",
        "confidence_threshold": 0.3
    }
    print("🔍 Object Detection Request:")
    print(f"   Text prompt: '{detect_request['text']}'")
    print(f"   Confidence threshold: {detect_request['confidence_threshold']}")
    print("   Expected: bounding boxes, scores, and labels")
    
    # Mock segmentation request
    segment_request = {
        "image": "base64_image_data",
        "text": "car on street",
        "confidence_threshold": 0.3,
        "include_masks": True
    }
    print("\n🎭 Object Segmentation Request:")
    print(f"   Text prompt: '{segment_request['text']}'")
    print(f"   Include masks: {segment_request['include_masks']}")
    print("   Expected: boxes + segmentation masks")

def test_qwen_vl_endpoints():
    """Test Qwen VL endpoints."""
    print("\n💬 Testing Qwen VL Service...")
    print("-" * 40)
    
    # Mock VQA request
    vqa_request = {
        "image": "base64_image_data",
        "question": "What color is the square?",
        "max_tokens": 256,
        "temperature": 0.1
    }
    print("❓ Visual Question Answering Request:")
    print(f"   Question: '{vqa_request['question']}'")
    print(f"   Max tokens: {vqa_request['max_tokens']}")
    print(f"   Temperature: {vqa_request['temperature']}")
    print("   Expected: text answer")
    
    # Mock streaming request
    print("\n📡 Streaming VQA Request:")
    print("   Question: 'Describe this image in detail'")
    print("   Stream: true")
    print("   Expected: Server-Sent Events stream")

def show_deployment_info():
    """Show how to deploy these services."""
    print("\n🚀 Deployment Information")
    print("=" * 50)
    
    print("\n1️⃣ Local Development:")
    print("   ./scripts/dev/run_local.sh clip")
    print("   ./scripts/dev/run_local.sh grounding_sam")
    print("   ./scripts/dev/run_local.sh qwen_vl")
    
    print("\n2️⃣ Docker Build:")
    print("   zoo build clip --gpu")
    print("   zoo build grounding_sam --gpu")
    print("   zoo build qwen_vl --gpu")
    
    print("\n3️⃣ Kubernetes Deploy:")
    print("   zoo deploy clip --replicas 3")
    print("   zoo deploy grounding_sam --gpus nvidia-l4")
    print("   zoo deploy qwen_vl --gpus nvidia-a100")
    
    print("\n4️⃣ Test Endpoints:")
    print("   curl http://localhost:8000/health")
    print("   curl http://localhost:8000/docs")
    print("   ./scripts/dev/test_api.sh http://localhost:8000 <model>")

if __name__ == "__main__":
    print("🦁 FastAPI Inference Menagerie - Endpoint Examples")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_image()
    print(f"\n📸 Created test image: {len(test_image)} bytes (base64)")
    
    # Test each service
    test_clip_endpoints()
    test_grounding_sam_endpoints()
    test_qwen_vl_endpoints()
    
    # Show deployment info
    show_deployment_info()
    
    print("\n✅ All endpoints documented!")