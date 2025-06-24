"""HTTP performance tests for CLIP service with actual network calls."""

import pytest
import time
import asyncio
import base64
import httpx
from pathlib import Path
from typing import List, Dict, Any
import threading
import uvicorn
from contextlib import asynccontextmanager

from fastapi import HTTPException
from services.clip.adapter import CLIPAdapter
from services.common.app import create_app


@pytest.mark.performance
class TestHTTPCLIPPerformance:
    """Test CLIP service with actual HTTP calls over the network."""
    
    @pytest.fixture(scope="class")
    async def clip_service_server(self):
        """Start CLIP service on a test port with real HTTP server."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/clip_http_perf_cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        # Create CLIP adapter
        adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await adapter.load_model()
        
        # Create FastAPI app with CLIP routes
        app = create_app(adapter)
        
        # Add CLIP-specific routes
        @app.post("/v1/clip/encode")
        async def encode_endpoint(payload: dict) -> dict:
            """Universal encoding endpoint for CLIP."""
            try:
                result = await adapter.predict(payload)
                return {"success": True, "result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Start server in background thread
        test_port = 8901
        server_thread = None
        server_ready = threading.Event()
        server_exception = None
        
        def run_server():
            nonlocal server_exception
            try:
                config = uvicorn.Config(
                    app, 
                    host="127.0.0.1", 
                    port=test_port,
                    log_level="warning"  # Reduce noise
                )
                server = uvicorn.Server(config)
                
                # Run server (this blocks) - don't signal ready until server is actually serving
                asyncio.run(server.serve())
            except Exception as e:
                server_exception = e
                server_ready.set()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        base_url = f"http://127.0.0.1:{test_port}"
        
        # Wait for server to be ready with health checks (no thread coordination)
        async with httpx.AsyncClient(timeout=10.0) as client:
            max_retries = 20  # Give plenty of time for CLIP model loading
            for attempt in range(max_retries):
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        print(f"CLIP server ready after {attempt+1} attempts")
                        break
                    else:
                        print(f"Health check attempt {attempt+1}: status {response.status_code}")
                except Exception as e:
                    print(f"Health check attempt {attempt+1} failed: {e}")
                    if attempt == max_retries - 1:
                        if server_exception:
                            pytest.skip(f"Server exception: {server_exception}")
                        else:
                            pytest.skip(f"Server failed to start after {max_retries} attempts: {e}")
                await asyncio.sleep(3)  # Wait 3 seconds between attempts
        
        yield base_url
        
        # Cleanup happens when test ends (daemon thread will die)
    
    @pytest.fixture(scope="class")
    def test_images(self):
        """Load test images for HTTP performance testing."""
        image_dir = Path(__file__).parent.parent / "data" / "images"
        images = {}
        
        for image_file in ["cat_office_typing.jpg", "dog_office_typing.jpg"]:
            image_path = image_dir / image_file
            with open(image_path, "rb") as f:
                image_data = f.read()
                # Include data URL prefix for HTTP requests
                b64_data = base64.b64encode(image_data).decode()
                images[image_file] = f"data:image/jpeg;base64,{b64_data}"
        
        return images
    
    @pytest.mark.asyncio
    async def test_http_text_encoding_latency(self, clip_service_server):
        """Test HTTP latency for text encoding with network calls."""
        base_url = clip_service_server
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Warm up
            await client.post(f"{base_url}/v1/clip/encode", json={
                "task": "encode_text",
                "texts": ["warm up"]
            })
            
            # Measure single text encoding over HTTP
            start_time = time.time()
            response = await client.post(f"{base_url}/v1/clip/encode", json={
                "task": "encode_text", 
                "texts": ["a cat"]
            })
            end_time = time.time()
            
            assert response.status_code == 200
            result = response.json()
            
            latency = end_time - start_time
            
            assert result["success"] == True
            assert result["result"]["count"] == 1
            assert len(result["result"]["embeddings"]) == 1
            
            print(f"HTTP Text encoding latency: {latency*1000:.3f}ms")
            
            # HTTP will be slower than direct adapter calls
            assert latency < 5.0  # Should be under 5 seconds for HTTP
    
    @pytest.mark.asyncio
    async def test_http_image_encoding_latency(self, clip_service_server, test_images):
        """Test HTTP latency for image encoding with network calls."""
        base_url = clip_service_server
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Warm up
            await client.post(f"{base_url}/v1/clip/encode", json={
                "task": "encode_image",
                "images": [test_images["cat_office_typing.jpg"]]
            })
            
            # Measure single image encoding over HTTP
            start_time = time.time()
            response = await client.post(f"{base_url}/v1/clip/encode", json={
                "task": "encode_image",
                "images": [test_images["cat_office_typing.jpg"]]
            })
            end_time = time.time()
            
            assert response.status_code == 200
            result = response.json()
            
            latency = end_time - start_time
            
            assert result["success"] == True
            assert result["result"]["count"] == 1
            assert len(result["result"]["embeddings"]) == 1
            
            print(f"HTTP Image encoding latency: {latency*1000:.3f}ms")
            
            # HTTP + base64 decoding will be slower
            assert latency < 10.0  # Should be under 10 seconds for HTTP
    
    @pytest.mark.asyncio
    async def test_http_similarity_computation(self, clip_service_server, test_images):
        """Test HTTP similarity computation with network calls."""
        base_url = clip_service_server
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Warm up
            await client.post(f"{base_url}/v1/clip/encode", json={
                "task": "similarity",
                "texts": ["a cat", "a dog"],
                "images": [test_images["cat_office_typing.jpg"]]
            })
            
            # Measure similarity computation over HTTP
            start_time = time.time()
            response = await client.post(f"{base_url}/v1/clip/encode", json={
                "task": "similarity",
                "texts": ["a cat", "a dog"],
                "images": [test_images["cat_office_typing.jpg"]]
            })
            end_time = time.time()
            
            assert response.status_code == 200
            result = response.json()
            
            latency = end_time - start_time
            
            assert "similarity_matrix" in result["result"]
            similarities = result["result"]["similarity_matrix"][0]
            assert similarities[0] > similarities[1]  # Cat image should match "a cat" better
            
            print(f"HTTP Similarity computation latency: {latency*1000:.3f}ms")
            print(f"Similarities - cat: {similarities[0]:.4f}, dog: {similarities[1]:.4f}")
    
    @pytest.mark.asyncio
    async def test_http_throughput_text_encoding(self, clip_service_server):
        """Test HTTP throughput for text encoding with real network calls."""
        base_url = clip_service_server
        texts = ["a cat", "a dog", "a bird", "a car", "a tree"]
        num_requests = 50
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Run individual HTTP requests to get latency distribution
            latencies = []
            start_time = time.time()
            
            for i in range(num_requests):
                payload = {
                    "task": "encode_text",
                    "texts": [texts[i % len(texts)]]
                }
                
                req_start = time.time()
                response = await client.post(f"{base_url}/v1/clip/encode", json=payload)
                req_end = time.time()
                
                assert response.status_code == 200
                result = response.json()
                
                latencies.append((req_end - req_start) * 1000)  # Convert to ms
                assert result["success"] == True
                assert result["result"]["count"] == 1
            
            end_time = time.time()
            total_time = end_time - start_time
            rps = num_requests / total_time
            
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            latencies_sorted = sorted(latencies)
            p50_latency = latencies_sorted[len(latencies_sorted) // 2]
            p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99_latency = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            
            print(f"\nHTTP CLIP Text Encoding Performance ({num_requests} requests):") 
            print(f"  Total time:    {total_time:.3f}s")
            print(f"  Throughput:    {rps:.1f} requests/second")
            print(f"  Latency stats:")
            print(f"    Average:     {avg_latency:.1f}ms")
            print(f"    Min:         {min_latency:.1f}ms")
            print(f"    Max:         {max_latency:.1f}ms")
            print(f"    P50:         {p50_latency:.1f}ms")
            print(f"    P95:         {p95_latency:.1f}ms")
            print(f"    P99:         {p99_latency:.1f}ms")
            
            # HTTP will have much lower throughput than direct calls
            assert rps > 0.5  # At least 0.5 RPS for HTTP CLIP
    
    @pytest.mark.asyncio
    async def test_http_cat_dog_classification_performance(self, clip_service_server, test_images):
        """Test HTTP cat vs dog classification with full network stack."""
        base_url = clip_service_server
        image_names = ["cat_office_typing.jpg", "dog_office_typing.jpg"]
        
        all_latencies = []
        correct_classifications = 0
        total_classifications = 0
        
        print(f"\nRunning HTTP classification tests (100 per image)...")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for image_name in image_names:
                print(f"\nTesting {image_name} over HTTP (100 iterations):")
                latencies = []
                correct = 0
                
                for i in range(100):
                    start = time.time()
                    response = await client.post(f"{base_url}/v1/clip/encode", json={
                        "task": "similarity",
                        "texts": ["a cat", "a dog"],
                        "images": [test_images[image_name]]
                    })
                    end = time.time()
                    
                    assert response.status_code == 200
                    result = response.json()
                    
                    latency = (end - start) * 1000  # Convert to ms
                    latencies.append(latency)
                    all_latencies.append(latency)
                    
                    similarities = result["similarity_matrix"][0]
                    is_cat = similarities[0] > similarities[1]
                    
                    # Check if classification is correct
                    if ("cat" in image_name and is_cat) or ("dog" in image_name and not is_cat):
                        correct += 1
                    
                    total_classifications += 1
                
                # Calculate statistics for this image
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                latencies_sorted = sorted(latencies)
                p50_latency = latencies_sorted[len(latencies_sorted) // 2]
                p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)]
                accuracy = (correct / 100) * 100
                
                print(f"  Accuracy:      {accuracy:.1f}% ({correct}/100)")
                print(f"  Latency stats:")
                print(f"    Average:     {avg_latency:.1f}ms")
                print(f"    Min:         {min_latency:.1f}ms")
                print(f"    Max:         {max_latency:.1f}ms")
                print(f"    P50:         {p50_latency:.1f}ms")
                print(f"    P95:         {p95_latency:.1f}ms")
                
                correct_classifications += correct
        
        # Overall statistics
        total_time = sum(all_latencies) / 1000  # Convert back to seconds
        overall_accuracy = (correct_classifications / total_classifications) * 100
        overall_avg_latency = sum(all_latencies) / len(all_latencies)
        
        # Overall latency distribution
        all_latencies_sorted = sorted(all_latencies)
        overall_p50 = all_latencies_sorted[len(all_latencies_sorted) // 2]
        overall_p95 = all_latencies_sorted[int(len(all_latencies_sorted) * 0.95)]
        overall_p99 = all_latencies_sorted[int(len(all_latencies_sorted) * 0.99)]
        
        print(f"\nOverall HTTP Cat vs Dog Classification Performance (200 total):")
        print(f"  Total time:    {total_time:.3f}s")
        print(f"  Throughput:    {200/total_time:.1f} classifications/second")
        print(f"  Accuracy:      {overall_accuracy:.1f}% ({correct_classifications}/200)")
        print(f"  Latency stats:")
        print(f"    Average:     {overall_avg_latency:.1f}ms")
        print(f"    P50:         {overall_p50:.1f}ms")
        print(f"    P95:         {overall_p95:.1f}ms")
        print(f"    P99:         {overall_p99:.1f}ms")
        
        # Verify high accuracy
        assert overall_accuracy > 90, f"Classification accuracy too low: {overall_accuracy:.1f}%"
    
    @pytest.mark.asyncio
    async def test_http_vs_direct_overhead_comparison(self, clip_service_server):
        """Compare HTTP vs direct adapter call overhead."""
        base_url = clip_service_server
        num_iterations = 10
        
        # Test direct adapter calls
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/clip_direct_comparison_cache")
        
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        direct_adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await direct_adapter.load_model()
        
        # Direct calls
        direct_latencies = []
        for i in range(num_iterations):
            start = time.time()
            result = await direct_adapter.predict({
                "task": "encode_text",
                "texts": [f"test text {i}"]
            })
            end = time.time()
            direct_latencies.append((end - start) * 1000)
            assert result["count"] == 1
        
        # HTTP calls
        http_latencies = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(num_iterations):
                start = time.time()
                response = await client.post(f"{base_url}/v1/clip/encode", json={
                    "task": "encode_text",
                    "texts": [f"test text {i}"]
                })
                end = time.time()
                http_latencies.append((end - start) * 1000)
                assert response.status_code == 200
                result = response.json()
                assert result["count"] == 1
        
        # Calculate averages
        direct_avg = sum(direct_latencies) / len(direct_latencies)
        http_avg = sum(http_latencies) / len(http_latencies)
        overhead = http_avg - direct_avg
        overhead_percentage = (overhead / direct_avg) * 100
        
        print(f"\nHTTP vs Direct Call Overhead Comparison ({num_iterations} iterations):")
        print(f"  Direct adapter avg:   {direct_avg:.1f}ms")
        print(f"  HTTP request avg:     {http_avg:.1f}ms")
        print(f"  HTTP overhead:        {overhead:.1f}ms ({overhead_percentage:.1f}% increase)")
        print(f"  HTTP is {http_avg/direct_avg:.1f}x slower than direct calls")
        
        # HTTP should have noticeable overhead
        assert http_avg > direct_avg, "HTTP should be slower than direct calls"
        assert overhead > 1.0, "HTTP overhead should be at least 1ms"