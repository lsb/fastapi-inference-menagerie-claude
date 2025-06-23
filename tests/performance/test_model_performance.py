"""Performance tests for model services."""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from services.clip.adapter import CLIPAdapter


@pytest.mark.performance
class TestModelPerformance:
    """Test model performance characteristics."""
    
    @pytest.fixture(scope="class")
    async def clip_adapter(self):
        """Real CLIP adapter for performance testing."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/performance_test_cache")
        adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await adapter.load_model()
        yield adapter
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, clip_adapter):
        """Test concurrent prediction performance."""
        
        # Test concurrent requests
        num_requests = 5  # Reduced for real model
        start_time = time.time()
        
        tasks = []
        for i in range(num_requests):
            payload = {
                "task": "encode_text",
                "texts": [f"test text {i}"]
            }
            tasks.append(clip_adapter.predict(payload))
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all requests completed
        assert len(results) == num_requests
        
        # Performance assertions - real CLIP model will be slower
        avg_time_per_request = total_time / num_requests
        assert avg_time_per_request < 2.0  # Allow up to 2s per request for CPU CLIP
        
        print(f"Processed {num_requests} requests in {total_time:.3f}s")
        print(f"Average time per request: {avg_time_per_request:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, clip_adapter):
        """Test that memory usage remains stable under load."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        # Initial memory measurement
        initial_memory = process.memory_info().rss
        
        # Run many predictions (reduced for real model)
        for i in range(10):
            payload = {
                "task": "encode_text", 
                "texts": [f"test text {i}"]
            }
            await clip_adapter.predict(payload)
            
            # Check memory every 5 iterations
            if i % 5 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't grow significantly (allow 100MB increase for real model)
                assert memory_increase < 100 * 1024 * 1024
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase after 10 predictions: {memory_increase / 1024 / 1024:.2f} MB")
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, clip_adapter):
        """Test that response times are consistent."""
        response_times = []
        
        # Run predictions and measure times (reduced for real model)
        for i in range(5):
            start = time.time()
            
            payload = {
                "task": "encode_text",
                "texts": [f"test text {i}"]
            }
            await clip_adapter.predict(payload)
            
            end = time.time()
            response_times.append(end - start)
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Response time consistency checks for real model
        assert max_time < avg_time * 3  # Max shouldn't be more than 3x average
        assert min_time > avg_time * 0.1  # Min shouldn't be less than 10% of average
        
        print(f"Response time stats - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
    
    def test_threading_safety(self, clip_adapter):
        """Test that adapter is thread-safe."""
        
        def worker(worker_id):
            """Worker function for threading test."""
            results = []
            for i in range(2):  # Reduced for real model
                # Run synchronous version for threading test
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                payload = {
                    "task": "encode_text",
                    "texts": [f"worker {worker_id} text {i}"]
                }
                result = loop.run_until_complete(clip_adapter.predict(payload))
                results.append(result)
                
                loop.close()
            
            return results
        
        # Run with multiple threads (reduced for real model)
        num_threads = 2
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for thread_id in range(num_threads):
                future = executor.submit(worker, thread_id)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in futures:
                thread_results = future.result()
                all_results.extend(thread_results)
        
        # Verify all requests completed successfully
        assert len(all_results) == num_threads * 2
        
        # Verify no corrupted results
        for result in all_results:
            assert "embeddings" in result
            assert result["count"] > 0


@pytest.mark.performance
@pytest.mark.slow
class TestLoadTesting:
    """Load testing for model services."""
    
    @pytest.fixture(scope="class")
    async def clip_adapter_load(self):
        """Real CLIP adapter for load testing."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/load_test_cache")
        adapter = CLIPAdapter(gcs_path=None, device="cpu")
        await adapter.load_model()
        yield adapter
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, clip_adapter_load):
        """Test performance under sustained load."""
        
        # Test parameters (reduced for real model)
        duration_seconds = 10
        requests_per_second = 2
        
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
                await clip_adapter_load.predict(payload)
                completed_requests += 1
            except Exception:
                errors += 1
        
        # Generate load
        request_id = 0
        while time.time() - start_time < duration_seconds:
            # Create batch of requests
            tasks = []
            for _ in range(requests_per_second):
                tasks.append(make_request(request_id))
                request_id += 1
            
            # Execute batch
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Wait for next second
            await asyncio.sleep(1.0)
        
        total_time = time.time() - start_time
        
        # Performance metrics
        actual_rps = completed_requests / total_time
        error_rate = errors / (completed_requests + errors) if (completed_requests + errors) > 0 else 0
        
        print(f"Load test results:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Completed requests: {completed_requests}")
        print(f"  Errors: {errors}")
        print(f"  Actual RPS: {actual_rps:.1f}")
        print(f"  Error rate: {error_rate:.2%}")
        
        # Assertions
        assert error_rate < 0.01  # Less than 1% error rate
        assert actual_rps > requests_per_second * 0.8  # At least 80% of target RPS
    
    @pytest.mark.asyncio
    async def test_burst_traffic(self, clip_adapter_load):
        """Test handling of burst traffic."""
        # Test burst of requests (reduced for real model)
        burst_size = 5
        start_time = time.time()
        
        tasks = []
        for i in range(burst_size):
            payload = {
                "task": "encode_text",
                "texts": [f"burst test text {i}"]
            }
            tasks.append(clip_adapter_load.predict(payload))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        burst_duration = end_time - start_time
        
        # Count successful requests
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        print(f"Burst test results:")
        print(f"  Burst size: {burst_size}")
        print(f"  Duration: {burst_duration:.3f}s")
        print(f"  Successful: {successful}")
        print(f"  RPS: {successful / burst_duration:.1f}")
        
        # Assertions for real model
        assert successful >= burst_size * 0.8  # At least 80% success rate for real model
        assert burst_duration < burst_size * 2.0  # Allow 2s per request for real model