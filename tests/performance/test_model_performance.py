"""Performance tests for model services."""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from tests.conftest import MockModelAdapter


@pytest.mark.performance
class TestModelPerformance:
    """Test model performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self):
        """Test concurrent prediction performance."""
        adapter = MockModelAdapter()
        
        # Test concurrent requests
        num_requests = 10
        start_time = time.time()
        
        tasks = []
        for i in range(num_requests):
            payload = {"test": f"data_{i}"}
            tasks.append(adapter.predict(payload))
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all requests completed
        assert len(results) == num_requests
        
        # Performance assertions
        avg_time_per_request = total_time / num_requests
        assert avg_time_per_request < 0.1  # Should be fast for mock
        
        print(f"Processed {num_requests} requests in {total_time:.3f}s")
        print(f"Average time per request: {avg_time_per_request:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that memory usage remains stable under load."""
        import psutil
        import os
        
        adapter = MockModelAdapter()
        process = psutil.Process(os.getpid())
        
        # Initial memory measurement
        initial_memory = process.memory_info().rss
        
        # Run many predictions
        for i in range(100):
            payload = {"test": f"data_{i}"}
            await adapter.predict(payload)
            
            # Check memory every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't grow significantly (allow 50MB increase)
                assert memory_increase < 50 * 1024 * 1024
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase after 100 predictions: {memory_increase / 1024 / 1024:.2f} MB")
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self):
        """Test that response times are consistent."""
        adapter = MockModelAdapter()
        
        response_times = []
        
        # Run predictions and measure times
        for i in range(20):
            start = time.time()
            
            payload = {"test": f"data_{i}"}
            await adapter.predict(payload)
            
            end = time.time()
            response_times.append(end - start)
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Response time consistency checks
        assert max_time < avg_time * 3  # Max shouldn't be more than 3x average
        assert min_time > avg_time * 0.1  # Min shouldn't be less than 10% of average
        
        print(f"Response time stats - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
    
    def test_threading_safety(self):
        """Test that adapter is thread-safe."""
        adapter = MockModelAdapter()
        
        def worker(worker_id):
            """Worker function for threading test."""
            results = []
            for i in range(10):
                # Run synchronous version for threading test
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                payload = {"worker": worker_id, "request": i}
                result = loop.run_until_complete(adapter.predict(payload))
                results.append(result)
                
                loop.close()
            
            return results
        
        # Run with multiple threads
        num_threads = 5
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
        assert len(all_results) == num_threads * 10
        
        # Verify no corrupted results
        for result in all_results:
            assert "result" in result
            assert result["result"] == "mock_prediction"


@pytest.mark.performance
@pytest.mark.slow
class TestLoadTesting:
    """Load testing for model services."""
    
    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """Test performance under sustained load."""
        adapter = MockModelAdapter()
        
        # Test parameters
        duration_seconds = 30
        requests_per_second = 10
        
        start_time = time.time()
        completed_requests = 0
        errors = 0
        
        async def make_request(request_id):
            """Make a single request."""
            nonlocal completed_requests, errors
            try:
                payload = {"load_test": True, "request_id": request_id}
                await adapter.predict(payload)
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
    async def test_burst_traffic(self):
        """Test handling of burst traffic."""
        adapter = MockModelAdapter()
        
        # Test burst of requests
        burst_size = 50
        start_time = time.time()
        
        tasks = []
        for i in range(burst_size):
            payload = {"burst_test": True, "request": i}
            tasks.append(adapter.predict(payload))
        
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
        
        # Assertions
        assert successful >= burst_size * 0.95  # At least 95% success rate
        assert burst_duration < burst_size * 0.1  # Should handle burst efficiently