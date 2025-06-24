"""Performance tests for CLIP service to measure ML model overhead."""

import pytest
import time
import asyncio
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from services.clip.adapter import CLIPAdapter


@pytest.mark.performance
class TestCLIPOverhead:
    """Test CLIP service to measure ML model inference overhead."""
    
    @pytest.fixture(scope="class")
    async def clip_adapter(self):
        """Create CLIP adapter for overhead testing."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/clip_performance_cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await adapter.load_model()
        yield adapter
    
    @pytest.fixture(scope="class")
    def test_images(self):
        """Load test images for performance testing."""
        image_dir = Path(__file__).parent.parent / "data" / "images"
        images = {}
        
        for image_file in ["cat_office_typing.jpg", "dog_office_typing.jpg"]:
            image_path = image_dir / image_file
            with open(image_path, "rb") as f:
                image_data = f.read()
                images[image_file] = base64.b64encode(image_data).decode()
        
        return images
    
    @pytest.mark.asyncio
    async def test_minimal_latency_text(self, clip_adapter):
        """Test minimal latency for text encoding."""
        # Warm up
        await clip_adapter.predict({
            "task": "encode_text",
            "texts": ["warm up"]
        })
        
        # Measure single text encoding
        start_time = time.time()
        result = await clip_adapter.predict({
            "task": "encode_text",
            "texts": ["a cat"]
        })
        end_time = time.time()
        
        latency = end_time - start_time
        
        assert result["count"] == 1
        assert len(result["embeddings"]) == 1
        
        print(f"Single text encoding latency: {latency*1000:.3f}ms")
        
        # CLIP will be much slower than is-odd
        assert latency < 1.0  # Should be under 1 second
    
    @pytest.mark.asyncio
    async def test_minimal_latency_image(self, clip_adapter, test_images):
        """Test minimal latency for image encoding."""
        # Warm up
        await clip_adapter.predict({
            "task": "encode_image",
            "images": [test_images["cat_office_typing.jpg"]]
        })
        
        # Measure single image encoding
        start_time = time.time()
        result = await clip_adapter.predict({
            "task": "encode_image",
            "images": [test_images["cat_office_typing.jpg"]]
        })
        end_time = time.time()
        
        latency = end_time - start_time
        
        assert result["count"] == 1
        assert len(result["embeddings"]) == 1
        
        print(f"Single image encoding latency: {latency*1000:.3f}ms")
        
        # Image encoding is typically slower than text
        assert latency < 2.0  # Should be under 2 seconds
    
    @pytest.mark.asyncio
    async def test_similarity_latency(self, clip_adapter, test_images):
        """Test latency for similarity computation."""
        # Warm up
        await clip_adapter.predict({
            "task": "similarity",
            "texts": ["a cat", "a dog"],
            "images": [test_images["cat_office_typing.jpg"]]
        })
        
        # Measure similarity computation
        start_time = time.time()
        result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["a cat", "a dog"],
            "images": [test_images["cat_office_typing.jpg"]]
        })
        end_time = time.time()
        
        latency = end_time - start_time
        
        assert "similarity_matrix" in result
        similarities = result["similarity_matrix"][0]
        assert similarities[0] > similarities[1]  # Cat image should match "a cat" better
        
        print(f"Similarity computation latency: {latency*1000:.3f}ms")
        print(f"Similarities - cat: {similarities[0]:.4f}, dog: {similarities[1]:.4f}")
    
    @pytest.mark.asyncio
    async def test_throughput_text_encoding(self, clip_adapter):
        """Test throughput for text encoding with statistical analysis."""
        texts = ["a cat", "a dog", "a bird", "a car", "a tree"]
        num_requests = 50  # Increased for better statistics
        
        # Run individual requests to get latency distribution
        latencies = []
        start_time = time.time()
        
        for i in range(num_requests):
            payload = {
                "task": "encode_text",
                "texts": [texts[i % len(texts)]]
            }
            
            req_start = time.time()
            result = await clip_adapter.predict(payload)
            req_end = time.time()
            
            latencies.append((req_end - req_start) * 1000)  # Convert to ms
            assert result["count"] == 1
        
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
        
        print(f"\nCLIP Text Encoding Performance ({num_requests} requests):")
        print(f"  Total time:    {total_time:.3f}s")
        print(f"  Throughput:    {rps:.1f} requests/second")
        print(f"  Latency stats:")
        print(f"    Average:     {avg_latency:.1f}ms")
        print(f"    Min:         {min_latency:.1f}ms")
        print(f"    Max:         {max_latency:.1f}ms")
        print(f"    P50:         {p50_latency:.1f}ms")
        print(f"    P95:         {p95_latency:.1f}ms")
        print(f"    P99:         {p99_latency:.1f}ms")
        
        # CLIP will have much lower throughput than is-odd
        assert rps > 1  # At least 1 RPS for CPU CLIP
    
    @pytest.mark.asyncio
    async def test_throughput_image_encoding(self, clip_adapter, test_images):
        """Test throughput for image encoding with statistical analysis."""
        images = list(test_images.values())
        num_requests = 500  # Increased for comprehensive statistics
        
        # Run individual requests to get latency distribution
        latencies = []
        start_time = time.time()
        
        for i in range(num_requests):
            payload = {
                "task": "encode_image",
                "images": [images[i % len(images)]]
            }
            
            req_start = time.time()
            result = await clip_adapter.predict(payload)
            req_end = time.time()
            
            latencies.append((req_end - req_start) * 1000)  # Convert to ms
            assert result["count"] == 1
        
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
        
        print(f"\nCLIP Image Encoding Performance ({num_requests} requests):")
        print(f"  Total time:    {total_time:.3f}s")
        print(f"  Throughput:    {rps:.1f} requests/second")
        print(f"  Latency stats:")
        print(f"    Average:     {avg_latency:.1f}ms")
        print(f"    Min:         {min_latency:.1f}ms")
        print(f"    Max:         {max_latency:.1f}ms")
        print(f"    P50:         {p50_latency:.1f}ms")
        print(f"    P95:         {p95_latency:.1f}ms")
        print(f"    P99:         {p99_latency:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_batch_vs_individual(self, clip_adapter):
        """Compare batch vs individual text encoding performance."""
        texts = ["a cat", "a dog", "a bird", "a car", "a tree"]
        
        # Test individual encoding
        start_time = time.time()
        individual_results = []
        for text in texts:
            result = await clip_adapter.predict({
                "task": "encode_text",
                "texts": [text]
            })
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Test batch encoding
        start_time = time.time()
        batch_result = await clip_adapter.predict({
            "task": "encode_text",
            "texts": texts
        })
        batch_time = time.time() - start_time
        
        # Verify results
        assert len(individual_results) == 5
        assert batch_result["count"] == 5
        
        individual_rps = 5 / individual_time
        batch_rps = 5 / batch_time
        speedup = batch_time / individual_time  # Lower is better
        
        print(f"Individual encoding: {individual_rps:.1f} items/s ({individual_time:.3f}s total)")
        print(f"Batch encoding: {batch_rps:.1f} items/s ({batch_time:.3f}s total)")
        print(f"Batch speedup: {1/speedup:.1f}x faster")
        
        # Batch should be significantly faster
        assert batch_time < individual_time
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, clip_adapter, test_images):
        """Test memory efficiency for CLIP inference."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run several predictions
        for i in range(5):
            # Text encoding
            await clip_adapter.predict({
                "task": "encode_text",
                "texts": [f"test text {i}"]
            })
            
            # Image encoding
            await clip_adapter.predict({
                "task": "encode_image",
                "images": [list(test_images.values())[i % 2]]
            })
            
            # Check memory
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory shouldn't grow significantly after model is loaded
            # Allow 200MB for CLIP model overhead
            assert memory_increase < 200 * 1024 * 1024
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase after 10 predictions: {memory_increase / 1024 / 1024:.2f} MB")
    
    @pytest.mark.asyncio
    async def test_cat_dog_classification_performance(self, clip_adapter, test_images):
        """Test performance of cat vs dog classification with comprehensive statistics."""
        # Test both images with 500 iterations for statistical significance
        image_names = [
            "cat_office_typing.jpg",
            "dog_office_typing.jpg"
        ]
        
        all_latencies = []
        correct_classifications = 0
        total_classifications = 0
        
        print(f"\nRunning 500 classification tests on each image...")
        
        for image_name in image_names:
            print(f"\nTesting {image_name} (500 iterations):")
            latencies = []
            correct = 0
            
            for i in range(500):
                start = time.time()
                result = await clip_adapter.predict({
                    "task": "similarity",
                    "texts": ["a cat", "a dog"],
                    "images": [test_images[image_name]]
                })
                end = time.time()
                
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
            p99_latency = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            accuracy = (correct / 500) * 100
            
            print(f"  Accuracy:      {accuracy:.1f}% ({correct}/500)")
            print(f"  Latency stats:")
            print(f"    Average:     {avg_latency:.1f}ms")
            print(f"    Min:         {min_latency:.1f}ms")
            print(f"    Max:         {max_latency:.1f}ms")
            print(f"    P50:         {p50_latency:.1f}ms")
            print(f"    P95:         {p95_latency:.1f}ms")
            print(f"    P99:         {p99_latency:.1f}ms")
            
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
        
        print(f"\nOverall Cat vs Dog Classification Performance (1000 total):")
        print(f"  Total time:    {total_time:.3f}s")
        print(f"  Throughput:    {1000/total_time:.1f} classifications/second")
        print(f"  Accuracy:      {overall_accuracy:.1f}% ({correct_classifications}/1000)")
        print(f"  Latency stats:")
        print(f"    Average:     {overall_avg_latency:.1f}ms")
        print(f"    P50:         {overall_p50:.1f}ms")
        print(f"    P95:         {overall_p95:.1f}ms")
        print(f"    P99:         {overall_p99:.1f}ms")
        
        # Verify high accuracy
        assert overall_accuracy > 90, f"Classification accuracy too low: {overall_accuracy:.1f}%"
    
    def test_threading_performance(self, clip_adapter):
        """Test CLIP performance with threading."""
        def worker(worker_id):
            """Worker function for threading test."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_time = time.time()
            
            # Each worker does 2 predictions
            for i in range(2):
                payload = {
                    "task": "encode_text",
                    "texts": [f"worker {worker_id} text {i}"]
                }
                result = loop.run_until_complete(clip_adapter.predict(payload))
                assert result["count"] == 1
            
            end_time = time.time()
            loop.close()
            
            return {
                "worker_id": worker_id,
                "time": end_time - start_time,
                "rps": 2 / (end_time - start_time)
            }
        
        # Test with different thread counts
        for num_threads in [1, 2]:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    future = executor.submit(worker, i)
                    futures.append(future)
                
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            total_requests = num_threads * 2
            overall_rps = total_requests / total_time
            
            print(f"\nThreads: {num_threads}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Overall RPS: {overall_rps:.1f}")
            for r in results:
                print(f"  Worker {r['worker_id']}: {r['rps']:.1f} RPS")


@pytest.mark.performance
class TestCLIPImageFormatOverhead:
    """Test overhead of different image formats for CLIP classification."""
    
    @pytest.fixture(scope="class")
    async def clip_adapter(self):
        """Create CLIP adapter for format testing."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/clip_format_test_cache")
        
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await adapter.load_model()
        yield adapter
    
    @pytest.fixture(scope="class")
    def format_test_images(self):
        """Load test images in different formats."""
        image_dir = Path(__file__).parent.parent / "data" / "images"
        images = {}
        
        # Test formats: high-quality JPG, low-quality JPG, BMP
        formats = [
            ("cat_office_typing.jpg", "High Quality JPG"),
            ("cat_office_typing_low_quality.jpg", "Low Quality JPG"),
            ("cat_office_typing.bmp", "BMP"),
            ("dog_office_typing.jpg", "High Quality JPG"),
            ("dog_office_typing_low_quality.jpg", "Low Quality JPG"),
            ("dog_office_typing.bmp", "BMP")
        ]
        
        for filename, format_name in formats:
            image_path = image_dir / filename
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    images[filename] = {
                        "data": base64.b64encode(image_data).decode(),
                        "format": format_name,
                        "size_kb": len(image_data) / 1024
                    }
            else:
                print(f"Warning: {image_path} not found")
        
        return images
    
    @pytest.mark.asyncio
    async def test_image_format_classification_performance(self, clip_adapter, format_test_images):
        """Test cat vs dog classification performance across different image formats."""
        
        # Test each format with 100 iterations for statistical significance
        format_results = {}
        
        print(f"\nTesting image format overhead for cat vs dog classification...")
        
        for filename, image_info in format_test_images.items():
            print(f"\nTesting {filename} ({image_info['format']}, {image_info['size_kb']:.1f}KB):")
            
            latencies = []
            correct = 0
            
            # Run 100 classifications per format
            for i in range(100):
                start = time.time()
                result = await clip_adapter.predict({
                    "task": "similarity",
                    "texts": ["a cat", "a dog"],
                    "images": [image_info["data"]]
                })
                end = time.time()
                
                latency = (end - start) * 1000  # Convert to ms
                latencies.append(latency)
                
                similarities = result["similarity_matrix"][0]
                is_cat = similarities[0] > similarities[1]
                
                # Check if classification is correct
                if ("cat" in filename and is_cat) or ("dog" in filename and not is_cat):
                    correct += 1
            
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            latencies_sorted = sorted(latencies)
            p50_latency = latencies_sorted[len(latencies_sorted) // 2]
            p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            accuracy = (correct / 100) * 100
            
            format_results[filename] = {
                "format": image_info["format"],
                "size_kb": image_info["size_kb"],
                "avg_latency": avg_latency,
                "p50_latency": p50_latency,
                "p95_latency": p95_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "accuracy": accuracy
            }
            
            print(f"  Size:          {image_info['size_kb']:.1f}KB")
            print(f"  Accuracy:      {accuracy:.1f}% ({correct}/100)")
            print(f"  Avg Latency:   {avg_latency:.1f}ms")
            print(f"  P50 Latency:   {p50_latency:.1f}ms")
            print(f"  P95 Latency:   {p95_latency:.1f}ms")
        
        # Compare formats
        print(f"\n" + "="*60)
        print(f"IMAGE FORMAT PERFORMANCE COMPARISON")
        print(f"="*60)
        
        # Group by animal for comparison
        cat_formats = {k: v for k, v in format_results.items() if "cat" in k}
        dog_formats = {k: v for k, v in format_results.items() if "dog" in k}
        
        for animal, formats in [("Cat", cat_formats), ("Dog", dog_formats)]:
            print(f"\n{animal} Images:")
            print(f"{'Format':<20} {'Size (KB)':<12} {'Avg Latency':<12} {'P95 Latency':<12} {'Accuracy':<10}")
            print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
            
            for filename, stats in formats.items():
                print(f"{stats['format']:<20} {stats['size_kb']:<12.1f} {stats['avg_latency']:<12.1f} {stats['p95_latency']:<12.1f} {stats['accuracy']:<10.1f}%")
        
        # Summary analysis
        print(f"\n" + "="*60)
        print(f"FORMAT ANALYSIS SUMMARY")
        print(f"="*60)
        
        # Compare high vs low quality JPG
        high_quality_latencies = []
        low_quality_latencies = []
        bmp_latencies = []
        
        for filename, stats in format_results.items():
            if "low_quality" in filename:
                low_quality_latencies.append(stats["avg_latency"])
            elif filename.endswith(".bmp"):
                bmp_latencies.append(stats["avg_latency"])
            elif filename.endswith(".jpg"):
                high_quality_latencies.append(stats["avg_latency"])
        
        if high_quality_latencies and low_quality_latencies:
            hq_avg = sum(high_quality_latencies) / len(high_quality_latencies)
            lq_avg = sum(low_quality_latencies) / len(low_quality_latencies)
            overhead_diff = abs(hq_avg - lq_avg)
            print(f"High Quality JPG avg: {hq_avg:.1f}ms")
            print(f"Low Quality JPG avg:  {lq_avg:.1f}ms")
            print(f"Quality difference:   {overhead_diff:.1f}ms ({overhead_diff/hq_avg*100:.1f}%)")
        
        if high_quality_latencies and bmp_latencies:
            hq_avg = sum(high_quality_latencies) / len(high_quality_latencies)
            bmp_avg = sum(bmp_latencies) / len(bmp_latencies)
            overhead_diff = abs(hq_avg - bmp_avg)
            print(f"High Quality JPG avg: {hq_avg:.1f}ms")
            print(f"BMP avg:              {bmp_avg:.1f}ms")
            print(f"Format difference:    {overhead_diff:.1f}ms ({overhead_diff/hq_avg*100:.1f}%)")
        
        # All formats should maintain high accuracy
        for filename, stats in format_results.items():
            assert stats["accuracy"] > 80, f"Low accuracy for {filename}: {stats['accuracy']:.1f}%"


@pytest.mark.performance
@pytest.mark.slow
class TestCLIPLoadTesting:
    """Load testing for CLIP service."""
    
    @pytest.fixture(scope="class")
    async def clip_adapter(self):
        """Create CLIP adapter for load testing."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/clip_load_test_cache")
        
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await adapter.load_model()
        yield adapter
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, clip_adapter):
        """Test CLIP performance under sustained load."""
        # Test parameters (heavily reduced for CLIP)
        duration_seconds = 10
        requests_per_second = 0.5  # One request every 2 seconds
        
        start_time = time.time()
        completed_requests = 0
        errors = 0
        
        async def make_request(request_id):
            """Make a single request."""
            nonlocal completed_requests, errors
            try:
                payload = {
                    "task": "encode_text",
                    "texts": [f"load test text {request_id}"]
                }
                await clip_adapter.predict(payload)
                completed_requests += 1
            except Exception as e:
                errors += 1
                print(f"Error in request {request_id}: {e}")
        
        # Generate load
        request_id = 0
        while time.time() - start_time < duration_seconds:
            # Create request
            task = asyncio.create_task(make_request(request_id))
            request_id += 1
            
            # Wait for next request time
            await asyncio.sleep(1.0 / requests_per_second)
        
        # Wait for any remaining requests
        await asyncio.sleep(2.0)
        
        total_time = time.time() - start_time
        
        # Performance metrics
        actual_rps = completed_requests / total_time
        error_rate = errors / (completed_requests + errors) if (completed_requests + errors) > 0 else 0
        
        print(f"\nCLIP Load test results:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Completed requests: {completed_requests}")
        print(f"  Errors: {errors}")
        print(f"  Actual RPS: {actual_rps:.2f}")
        print(f"  Error rate: {error_rate:.2%}")
        
        # Assertions (relaxed for CLIP)
        assert error_rate < 0.1  # Less than 10% error rate
        assert actual_rps > requests_per_second * 0.5  # At least 50% of target RPS