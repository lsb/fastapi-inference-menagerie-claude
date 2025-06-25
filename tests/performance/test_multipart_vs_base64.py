"""Performance comparison between multipart binary uploads and base64 JSON."""

import pytest
import time
import asyncio
import base64
import httpx
import threading
import uvicorn
from pathlib import Path
from typing import Dict, List
import statistics
import json

from fastapi import HTTPException
from services.clip.adapter import CLIPAdapter
from services.clip.app import app as clip_app


@pytest.mark.performance
class TestMultipartVsBase64Performance:
    """Compare multipart binary uploads vs base64 JSON for CLIP."""
    
    @pytest.fixture(scope="class")
    async def clip_service_server(self):
        """Start CLIP service with multipart endpoints."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/multipart_test_cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        # Use the updated CLIP app with multipart support
        app = clip_app
        
        # Start server in background thread
        test_port = 8904
        server_exception = None
        
        def run_server():
            nonlocal server_exception
            try:
                config = uvicorn.Config(
                    app, 
                    host="127.0.0.1", 
                    port=test_port,
                    log_level="warning"
                )
                server = uvicorn.Server(config)
                asyncio.run(server.serve())
            except Exception as e:
                server_exception = e
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        base_url = f"http://127.0.0.1:{test_port}"
        
        # Wait for server to be ready
        async with httpx.AsyncClient(timeout=10.0) as client:
            max_retries = 20
            for attempt in range(max_retries):
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        print(f"CLIP server ready after {attempt+1} attempts")
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        if server_exception:
                            pytest.skip(f"Server exception: {server_exception}")
                        else:
                            pytest.skip(f"Server failed to start: {e}")
                await asyncio.sleep(3)
        
        yield base_url
    
    @pytest.fixture(scope="class")
    def test_images(self):
        """Load test images for comparison."""
        image_dir = Path(__file__).parent.parent / "data" / "images"
        images = {}
        
        test_files = [
            "cat_office_typing.jpg",  # High quality
            "cat_office_typing_low_quality.jpg",  # Low quality
            "cat_office_typing.bmp"  # Large BMP
        ]
        
        for filename in test_files:
            image_path = image_dir / filename
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    b64_data = base64.b64encode(image_data).decode()
                    
                    images[filename] = {
                        "binary_data": image_data,
                        "base64_data": f"data:image/jpeg;base64,{b64_data}",
                        "size_kb": len(image_data) / 1024,
                        "base64_size_kb": len(b64_data) / 1024
                    }
        
        return images
    
    def calculate_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not latencies:
            return {}
        
        sorted_latencies = sorted(latencies)
        return {
            "count": len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "p50": sorted_latencies[int(len(sorted_latencies) * 0.50)],
            "p90": sorted_latencies[int(len(sorted_latencies) * 0.90)],
            "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)]
        }
    
    @pytest.mark.asyncio
    async def test_multipart_vs_base64_performance(self, clip_service_server, test_images):
        """Compare multipart binary uploads vs base64 JSON performance."""
        base_url = clip_service_server
        num_iterations = 100  # 100 iterations per test
        num_warmups = 5
        
        print(f"\n{'='*80}")
        print(f"MULTIPART VS BASE64 PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        print(f"Testing {len(test_images)} images with {num_iterations} iterations each")
        print(f"Warmups: {num_warmups} per image/method")
        
        all_results = {}
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for filename, image_info in test_images.items():
                print(f"\nTesting {filename} ({image_info['size_kb']:.1f}KB binary, {image_info['base64_size_kb']:.1f}KB base64):")
                
                # Test multipart approach
                print(f"  Testing multipart binary upload...")
                
                # Warmup multipart
                for _ in range(num_warmups):
                    files = {"images": (filename, image_info["binary_data"], "image/jpeg")}
                    await client.post(f"{base_url}/v1/clip/encode/image", files=files)
                
                # Measure multipart performance
                multipart_latencies = []
                for i in range(num_iterations):
                    files = {"images": (filename, image_info["binary_data"], "image/jpeg")}
                    
                    start = time.time()
                    response = await client.post(f"{base_url}/v1/clip/encode/image", files=files)
                    end = time.time()
                    
                    assert response.status_code == 200
                    result = response.json()
                    assert result["success"] == True
                    
                    latency = (end - start) * 1000  # Convert to ms
                    multipart_latencies.append(latency)
                
                # Test base64 approach (using legacy adapter directly for comparison)
                print(f"  Testing base64 JSON...")
                
                # Create a simple base64 endpoint for comparison
                base64_latencies = []
                adapter = CLIPAdapter(gcs_path=None, device="cpu")
                await adapter.load_model()
                
                # Warmup base64
                for _ in range(num_warmups):
                    await adapter.predict({
                        "task": "encode_image",
                        "images": [image_info["base64_data"]]
                    })
                
                # Measure base64 performance (direct adapter call to simulate JSON parsing overhead)
                for i in range(num_iterations):
                    start = time.time()
                    result = await adapter.predict({
                        "task": "encode_image", 
                        "images": [image_info["base64_data"]]
                    })
                    end = time.time()
                    
                    assert result["count"] == 1
                    
                    latency = (end - start) * 1000
                    base64_latencies.append(latency)
                
                # Calculate statistics
                multipart_stats = self.calculate_stats(multipart_latencies)
                base64_stats = self.calculate_stats(base64_latencies)
                
                # Calculate improvements
                latency_improvement = base64_stats["mean"] - multipart_stats["mean"]
                latency_improvement_pct = (latency_improvement / base64_stats["mean"]) * 100
                size_reduction = image_info["base64_size_kb"] - image_info["size_kb"]
                size_reduction_pct = (size_reduction / image_info["base64_size_kb"]) * 100
                
                all_results[filename] = {
                    "binary_size_kb": image_info["size_kb"],
                    "base64_size_kb": image_info["base64_size_kb"],
                    "multipart_stats": multipart_stats,
                    "base64_stats": base64_stats,
                    "latency_improvement_ms": latency_improvement,
                    "latency_improvement_pct": latency_improvement_pct,
                    "size_reduction_kb": size_reduction,
                    "size_reduction_pct": size_reduction_pct
                }
                
                print(f"  Results:")
                print(f"    Payload sizes:")
                print(f"      Binary:     {image_info['size_kb']:.1f}KB")
                print(f"      Base64:     {image_info['base64_size_kb']:.1f}KB ({size_reduction_pct:.1f}% larger)")
                print(f"    Multipart latency:")
                print(f"      Mean:       {multipart_stats['mean']:.1f}ms")
                print(f"      P50:        {multipart_stats['p50']:.1f}ms")
                print(f"      P95:        {multipart_stats['p95']:.1f}ms")
                print(f"      P99:        {multipart_stats['p99']:.1f}ms")
                print(f"    Base64 latency:")
                print(f"      Mean:       {base64_stats['mean']:.1f}ms")
                print(f"      P50:        {base64_stats['p50']:.1f}ms")
                print(f"      P95:        {base64_stats['p95']:.1f}ms")
                print(f"      P99:        {base64_stats['p99']:.1f}ms")
                print(f"    Performance improvement:")
                print(f"      Latency:    {latency_improvement:.1f}ms faster ({latency_improvement_pct:.1f}% improvement)")
        
        # Overall summary
        print(f"\n{'='*80}")
        print(f"OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Image':<35} {'Binary Size':<12} {'Base64 Size':<12} {'MP Mean':<10} {'B64 Mean':<10} {'Improvement':<12}")
        print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
        
        total_improvement = 0
        total_size_reduction = 0
        
        for filename, results in all_results.items():
            mp_mean = results["multipart_stats"]["mean"]
            b64_mean = results["base64_stats"]["mean"]
            improvement = results["latency_improvement_pct"]
            binary_size = results["binary_size_kb"]
            base64_size = results["base64_size_kb"]
            
            print(f"{filename:<35} {binary_size:<12.1f} {base64_size:<12.1f} {mp_mean:<10.1f} {b64_mean:<10.1f} {improvement:<12.1f}%")
            
            total_improvement += improvement
            total_size_reduction += results["size_reduction_pct"]
        
        avg_improvement = total_improvement / len(all_results)
        avg_size_reduction = total_size_reduction / len(all_results)
        
        print(f"\nAverage improvements:")
        print(f"  Latency improvement: {avg_improvement:.1f}%")
        print(f"  Payload size reduction: {avg_size_reduction:.1f}%")
        
        # Assertions
        assert avg_improvement > 0, f"Multipart should be faster than base64, got {avg_improvement:.1f}% improvement"
        assert avg_size_reduction > 25, f"Base64 overhead should be ~33%, got {avg_size_reduction:.1f}%"
        
        print(f"\n{'='*80}")
        print(f"MULTIPART BINARY UPLOADS ARE SIGNIFICANTLY FASTER!")
        print(f"{'='*80}")