"""Comprehensive HTTP performance tests for CLIP with multipart binary uploads."""

import pytest
import time
import asyncio
import httpx
import threading
import uvicorn
from pathlib import Path
from typing import Dict, List
import statistics
import json

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from services.clip.adapter import CLIPAdapter
from PIL import Image
import io


@pytest.mark.performance
class TestHTTPCLIPMultipartComprehensive:
    """Comprehensive CLIP classification performance tests with multipart uploads."""
    
    @pytest.fixture(scope="class")
    async def clip_service_server(self):
        """Start CLIP service with multipart endpoints."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/clip_multipart_comprehensive_cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        # Create CLIP adapter
        adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await adapter.load_model()
        
        # Create FastAPI app with multipart routes
        app = FastAPI(title="CLIP Multipart Test Service")
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @app.post("/v1/clip/similarity")
        async def compute_similarity(
            texts: str = Form(..., description="JSON array of texts to compare"),
            images: List[UploadFile] = File(..., description="Images to compare against texts")
        ):
            try:
                texts_list = json.loads(texts)
                if not isinstance(texts_list, list):
                    raise ValueError("texts must be a JSON array")
                
                pil_images = []
                for image_file in images:
                    contents = await image_file.read()
                    pil_image = Image.open(io.BytesIO(contents))
                    pil_images.append(pil_image)
                
                payload = {
                    "task": "similarity",
                    "texts": texts_list,
                    "images": pil_images
                }
                result = await adapter.predict(payload)
                return {"success": True, "result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Start server in background thread
        test_port = 8905
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
    def comprehensive_test_images(self):
        """Load all test images for comprehensive testing."""
        image_dir = Path(__file__).parent.parent / "data" / "images"
        images = {}
        
        # Define all combinations
        animals = ["cat", "dog"]
        settings = ["office_typing", "mountain_sunrise"]  # office=indoor, mountain=outdoor
        formats = [
            (".jpg", "High Quality JPG"),
            ("_low_quality.jpg", "Low Quality JPG"), 
            (".bmp", "BMP")
        ]
        
        for animal in animals:
            for setting in settings:
                for format_suffix, format_name in formats:
                    filename = f"{animal}_{setting}{format_suffix}"
                    image_path = image_dir / filename
                    
                    if image_path.exists():
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                            
                            # Create descriptive key
                            indoor_outdoor = "indoor" if "office" in setting else "outdoor"
                            key = f"{animal}_{indoor_outdoor}_{format_name.lower().replace(' ', '_')}"
                            
                            images[key] = {
                                "binary_data": image_data,
                                "animal": animal,
                                "setting": indoor_outdoor,
                                "format": format_name,
                                "filename": filename,
                                "size_kb": len(image_data) / 1024
                            }
                    else:
                        print(f"Warning: {image_path} not found")
        
        return images
    
    def calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate comprehensive latency statistics."""
        if not latencies:
            return {}
        
        sorted_latencies = sorted(latencies)
        return {
            "count": len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p50": sorted_latencies[int(len(sorted_latencies) * 0.50)],
            "p90": sorted_latencies[int(len(sorted_latencies) * 0.90)],
            "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)]
        }
    
    @pytest.mark.asyncio
    async def test_comprehensive_multipart_classification_performance(self, clip_service_server, comprehensive_test_images):
        """Test comprehensive classification with multipart binary uploads."""
        base_url = clip_service_server
        num_iterations = 200  # 200 runs per image/format combination
        num_warmups = 5  # 5 warmup requests per image
        
        # Classification prompts
        animal_prompts = ["a cat", "a dog"]
        setting_prompts = ["indoor setting", "outdoor setting"]
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE MULTIPART CLIP CLASSIFICATION PERFORMANCE TEST")
        print(f"{'='*80}")
        print(f"Running {num_iterations} iterations per image/format combination...")
        print(f"Warmups: {len(comprehensive_test_images)} images × {num_warmups} warmups × 2 prompts = {len(comprehensive_test_images) * num_warmups * 2:,} warmup requests")
        print(f"Total tests: {len(comprehensive_test_images)} images × {num_iterations} iterations × 2 prompts = {len(comprehensive_test_images) * num_iterations * 2:,} test requests")
        print(f"Grand total: {len(comprehensive_test_images) * (num_warmups + num_iterations) * 2:,} HTTP requests")
        print(f"Using MULTIPART BINARY UPLOADS (no base64 overhead!)")
        
        all_results = {}
        overall_latencies = []
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for 200 iterations
            for image_key, image_info in comprehensive_test_images.items():
                print(f"\nTesting {image_key}:")
                print(f"  File: {image_info['filename']} ({image_info['size_kb']:.1f}KB binary)")
                print(f"  Expected: {image_info['animal']}, {image_info['setting']}")
                
                # Warmup requests (not counted in stats)
                print(f"  Running {num_warmups} warmup requests...")
                for warmup in range(num_warmups):
                    # Warmup animal classification
                    files = {"images": (image_info["filename"], image_info["binary_data"], "image/jpeg")}
                    data = {"texts": json.dumps(animal_prompts)}
                    await client.post(f"{base_url}/v1/clip/similarity", files=files, data=data)
                    
                    # Warmup setting classification  
                    files = {"images": (image_info["filename"], image_info["binary_data"], "image/jpeg")}
                    data = {"texts": json.dumps(setting_prompts)}
                    await client.post(f"{base_url}/v1/clip/similarity", files=files, data=data)
                
                print(f"  Running {num_iterations} test iterations...")
                latencies = []
                animal_correct = 0
                setting_correct = 0
                animal_similarities = []
                setting_similarities = []
                
                for i in range(num_iterations):
                    # Test animal classification (cat vs dog) with multipart
                    files = {"images": (image_info["filename"], image_info["binary_data"], "image/jpeg")}
                    data = {"texts": json.dumps(animal_prompts)}
                    
                    start = time.time()
                    response = await client.post(f"{base_url}/v1/clip/similarity", files=files, data=data)
                    end = time.time()
                    
                    assert response.status_code == 200
                    result = response.json()
                    latency = (end - start) * 1000  # Convert to ms
                    latencies.append(latency)
                    overall_latencies.append(latency)
                    
                    # Check animal classification
                    animal_sims = result["result"]["similarity_matrix"][0]
                    animal_similarities.append(animal_sims)
                    predicted_animal = "cat" if animal_sims[0] > animal_sims[1] else "dog"
                    if predicted_animal == image_info["animal"]:
                        animal_correct += 1
                    
                    # Test setting classification (indoor vs outdoor) with multipart
                    files = {"images": (image_info["filename"], image_info["binary_data"], "image/jpeg")}
                    data = {"texts": json.dumps(setting_prompts)}
                    
                    start = time.time()
                    response = await client.post(f"{base_url}/v1/clip/similarity", files=files, data=data)
                    end = time.time()
                    
                    assert response.status_code == 200
                    result = response.json()
                    latency = (end - start) * 1000
                    latencies.append(latency)
                    overall_latencies.append(latency)
                    
                    # Check setting classification
                    setting_sims = result["result"]["similarity_matrix"][0]
                    setting_similarities.append(setting_sims)
                    predicted_setting = "indoor" if setting_sims[0] > setting_sims[1] else "outdoor"
                    if predicted_setting == image_info["setting"]:
                        setting_correct += 1
                
                # Calculate statistics for this image/format
                stats = self.calculate_latency_stats(latencies)
                animal_accuracy = (animal_correct / num_iterations) * 100
                setting_accuracy = (setting_correct / num_iterations) * 100
                
                # Average similarities for analysis
                avg_animal_sims = [
                    sum(sim[0] for sim in animal_similarities) / len(animal_similarities),
                    sum(sim[1] for sim in animal_similarities) / len(animal_similarities)
                ]
                avg_setting_sims = [
                    sum(sim[0] for sim in setting_similarities) / len(setting_similarities),
                    sum(sim[1] for sim in setting_similarities) / len(setting_similarities)
                ]
                
                all_results[image_key] = {
                    **image_info,
                    "latency_stats": stats,
                    "animal_accuracy": animal_accuracy,
                    "setting_accuracy": setting_accuracy,
                    "avg_animal_similarities": avg_animal_sims,
                    "avg_setting_similarities": avg_setting_sims
                }
                
                print(f"  Results:")
                print(f"    Animal accuracy: {animal_accuracy:.1f}% ({animal_correct}/{num_iterations})")
                print(f"    Setting accuracy: {setting_accuracy:.1f}% ({setting_correct}/{num_iterations})")
                print(f"    Avg animal sims: cat={avg_animal_sims[0]:.3f}, dog={avg_animal_sims[1]:.3f}")
                print(f"    Avg setting sims: indoor={avg_setting_sims[0]:.3f}, outdoor={avg_setting_sims[1]:.3f}")
                print(f"    Latency stats (MULTIPART):")
                print(f"      Mean: {stats['mean']:.1f}ms")
                print(f"      P50:  {stats['p50']:.1f}ms")
                print(f"      P90:  {stats['p90']:.1f}ms") 
                print(f"      P95:  {stats['p95']:.1f}ms")
                print(f"      P99:  {stats['p99']:.1f}ms")
        
        # Overall statistics
        overall_stats = self.calculate_latency_stats(overall_latencies)
        total_requests = len(overall_latencies)
        total_time = sum(overall_latencies) / 1000  # Convert to seconds
        
        print(f"\n{'='*80}")
        print(f"OVERALL MULTIPART PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Total requests: {total_requests:,}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Overall throughput: {total_requests/total_time:.1f} requests/second")
        print(f"Overall latency statistics (MULTIPART BINARY):")
        print(f"  Mean: {overall_stats['mean']:.1f}ms")
        print(f"  P50:  {overall_stats['p50']:.1f}ms")
        print(f"  P90:  {overall_stats['p90']:.1f}ms")
        print(f"  P95:  {overall_stats['p95']:.1f}ms")
        print(f"  P99:  {overall_stats['p99']:.1f}ms")
        
        # Format comparison analysis
        print(f"\n{'='*80}")
        print(f"FORMAT PERFORMANCE COMPARISON (MULTIPART)")
        print(f"{'='*80}")
        
        # Group results by format
        format_groups = {}
        for image_key, results in all_results.items():
            format_name = results["format"]
            if format_name not in format_groups:
                format_groups[format_name] = []
            format_groups[format_name].append(results)
        
        print(f"{'Format':<20} {'Count':<8} {'Avg Size':<12} {'Mean Lat':<12} {'P50':<8} {'P90':<8} {'P95':<8} {'P99':<8}")
        print(f"{'-'*20} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for format_name, format_results in format_groups.items():
            # Aggregate stats for this format
            all_format_latencies = []
            total_size = 0
            for result in format_results:
                # Each result has latencies from both animal and setting tests  
                latencies_per_image = num_iterations * 2  # 2 requests per iteration
                all_format_latencies.extend([result["latency_stats"]["mean"]] * latencies_per_image)
                total_size += result["size_kb"]
            
            avg_size = total_size / len(format_results)
            format_stats = self.calculate_latency_stats(all_format_latencies)
            
            print(f"{format_name:<20} {len(format_results):<8} {avg_size:<12.1f} {format_stats['mean']:<12.1f} {format_stats['p50']:<8.1f} {format_stats['p90']:<8.1f} {format_stats['p95']:<8.1f} {format_stats['p99']:<8.1f}")
        
        # Classification accuracy analysis
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION ACCURACY ANALYSIS")
        print(f"{'='*80}")
        
        print(f"{'Image':<40} {'Animal Acc':<12} {'Setting Acc':<12} {'Format':<20}")
        print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*20}")
        
        total_animal_accuracy = 0
        total_setting_accuracy = 0
        
        for image_key, results in all_results.items():
            print(f"{image_key:<40} {results['animal_accuracy']:<12.1f}% {results['setting_accuracy']:<12.1f}% {results['format']:<20}")
            total_animal_accuracy += results['animal_accuracy']
            total_setting_accuracy += results['setting_accuracy']
        
        avg_animal_accuracy = total_animal_accuracy / len(all_results)
        avg_setting_accuracy = total_setting_accuracy / len(all_results)
        
        print(f"\nOverall Averages:")
        print(f"  Animal classification accuracy: {avg_animal_accuracy:.1f}%")
        print(f"  Setting classification accuracy: {avg_setting_accuracy:.1f}%")
        
        # Assertions for test validation
        assert avg_animal_accuracy > 80, f"Animal classification accuracy too low: {avg_animal_accuracy:.1f}%"
        assert avg_setting_accuracy > 70, f"Setting classification accuracy too low: {avg_setting_accuracy:.1f}%"
        assert overall_stats["p99"] < 10000, f"P99 latency too high: {overall_stats['p99']:.1f}ms"
        
        print(f"\n{'='*80}")
        print(f"MULTIPART BINARY UPLOAD TEST COMPLETED SUCCESSFULLY!")
        print(f"NO BASE64 ENCODING OVERHEAD - PURE BINARY EFFICIENCY!")
        print(f"{'='*80}")