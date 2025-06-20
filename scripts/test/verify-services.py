#!/usr/bin/env python3
"""Verify all services can be imported and initialized."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_service_imports():
    """Test that all services can be imported."""
    print("🔍 Verifying service imports...")
    
    services = [
        ("CLIP", "services.clip.app", "app"),
        ("Grounding DINO + SAM2", "services.grounding_sam.app", "app"),
        ("Qwen VL", "services.qwen_vl.app", "app")
    ]
    
    all_passed = True
    
    for name, module_path, app_name in services:
        try:
            module = __import__(module_path, fromlist=[app_name])
            app = getattr(module, app_name)
            print(f"✅ {name} - Successfully imported")
            print(f"   App title: {app.title}")
            print(f"   Version: {app.version}")
        except Exception as e:
            print(f"❌ {name} - Failed to import: {e}")
            all_passed = False
    
    return all_passed

def test_adapter_imports():
    """Test that all adapters can be imported."""
    print("\n🔍 Verifying adapter imports...")
    
    adapters = [
        ("CLIP Adapter", "services.clip.adapter", "CLIPAdapter"),
        ("Grounding SAM Adapter", "services.grounding_sam.adapter", "GroundingSAMAdapter"),
        ("Qwen VL Adapter", "services.qwen_vl.adapter", "QwenVLAdapter")
    ]
    
    all_passed = True
    
    for name, module_path, class_name in adapters:
        try:
            module = __import__(module_path, fromlist=[class_name])
            adapter_class = getattr(module, class_name)
            print(f"✅ {name} - Successfully imported")
            print(f"   Class: {adapter_class.__name__}")
        except Exception as e:
            print(f"❌ {name} - Failed to import: {e}")
            all_passed = False
    
    return all_passed

def test_api_endpoints():
    """Test that API endpoints are defined."""
    print("\n🔍 Verifying API endpoints...")
    
    from services.clip.app import app as clip_app
    from services.grounding_sam.app import app as grounding_app
    from services.qwen_vl.app import app as qwen_app
    
    apps = [
        ("CLIP", clip_app, ["/v1/clip/encode/text", "/v1/clip/encode/image", "/v1/clip/similarity"]),
        ("Grounding DINO + SAM2", grounding_app, ["/v1/ground/segment", "/v1/ground/detect"]),
        ("Qwen VL", qwen_app, ["/v1/vqa/ask", "/v1/vqa/chat"])
    ]
    
    all_passed = True
    
    for name, app, expected_endpoints in apps:
        print(f"\n{name} endpoints:")
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        for endpoint in expected_endpoints:
            if endpoint in routes:
                print(f"  ✅ {endpoint}")
            else:
                print(f"  ❌ {endpoint} - Not found")
                all_passed = False
        
        # Show other routes
        other_routes = [r for r in routes if r not in expected_endpoints and not r.startswith('/openapi')]
        if other_routes:
            print(f"  📋 Other routes: {', '.join(other_routes[:5])}")
    
    return all_passed

if __name__ == "__main__":
    print("🦁 FastAPI Inference Menagerie - Service Verification")
    print("=" * 50)
    
    # Set minimal environment variables
    os.environ["DEVICE"] = "cpu"
    os.environ["CACHE_DIR"] = "/tmp/test_cache"
    
    results = []
    results.append(test_service_imports())
    results.append(test_adapter_imports())
    results.append(test_api_endpoints())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All verifications passed!")
        sys.exit(0)
    else:
        print("❌ Some verifications failed!")
        sys.exit(1)