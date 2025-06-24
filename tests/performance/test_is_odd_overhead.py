"""Performance tests for is-odd service to measure FastAPI overhead."""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from services.is_odd.adapter import IsOddAdapter


@pytest.mark.performance
class TestIsOddOverhead:
    """Test is-odd service to measure pure FastAPI overhead."""
    
    @pytest.fixture(scope="class")
    async def is_odd_adapter(self):
        """Create is-odd adapter for overhead testing."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/performance_test_cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = IsOddAdapter()
        await adapter.load_model()
        yield adapter
    
    @pytest.mark.asyncio
    async def test_minimal_latency(self, is_odd_adapter):
        """Test minimal latency for simple computation."""
        # Warm up
        await is_odd_adapter.predict({"number": 1})
        
        # Measure single request latency
        start_time = time.time()
        result = await is_odd_adapter.predict({"number": 42})
        end_time = time.time()
        
        latency = end_time - start_time
        
        assert result["is_odd"] is False
        assert latency < 0.001  # Should be sub-millisecond
        
        print(f"Single request latency: {latency*1000:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_throughput_baseline(self, is_odd_adapter):
        """Test maximum throughput for simple computation."""
        num_requests = 1000
        start_time = time.time()
        
        tasks = []
        for i in range(num_requests):
            payload = {"number": i}
            tasks.append(is_odd_adapter.predict(payload))
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        rps = num_requests / total_time
        
        assert len(results) == num_requests
        
        # Verify some results are correct
        assert results[0]["is_odd"] is False  # 0 is even
        assert results[1]["is_odd"] is True   # 1 is odd
        assert results[2]["is_odd"] is False  # 2 is even
        
        print(f"Processed {num_requests} requests in {total_time:.3f}s")
        print(f"Throughput: {rps:.0f} requests/second")
        print(f"Average latency: {(total_time/num_requests)*1000:.3f}ms")
        
        # Should achieve very high throughput for simple computation
        assert rps > 10000  # Expect >10k RPS for simple math
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, is_odd_adapter):
        """Test memory efficiency for many requests."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run many requests
        for batch in range(10):
            tasks = []
            for i in range(100):
                payload = {"number": batch * 100 + i}
                tasks.append(is_odd_adapter.predict(payload))
            
            await asyncio.gather(*tasks)
            
            # Check memory every batch
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory should not grow significantly for stateless computation
            assert memory_increase < 10 * 1024 * 1024  # Less than 10MB increase
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase after 1000 requests: {memory_increase / 1024 / 1024:.2f} MB")
    
    @pytest.mark.asyncio
    async def test_concurrency_scaling(self, is_odd_adapter):
        """Test how concurrency affects performance."""
        request_counts = [1, 10, 100, 500]
        
        for num_requests in request_counts:
            start_time = time.time()
            
            tasks = []
            for i in range(num_requests):
                payload = {"number": i}
                tasks.append(is_odd_adapter.predict(payload))
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            rps = num_requests / total_time
            avg_latency = (total_time / num_requests) * 1000
            
            assert len(results) == num_requests
            
            print(f"Requests: {num_requests:3d}, Time: {total_time:.3f}s, "
                  f"RPS: {rps:8.0f}, Avg Latency: {avg_latency:.3f}ms")
    
    def test_cpu_utilization(self, is_odd_adapter):
        """Test CPU utilization with threading."""
        def worker():
            """Worker function for CPU test."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            requests_per_worker = 100
            start_time = time.time()
            
            for i in range(requests_per_worker):
                payload = {"number": i}
                result = loop.run_until_complete(is_odd_adapter.predict(payload))
                assert "is_odd" in result
            
            end_time = time.time()
            loop.close()
            
            return {
                "requests": requests_per_worker,
                "time": end_time - start_time,
                "rps": requests_per_worker / (end_time - start_time)
            }
        
        # Test with different numbers of threads
        thread_counts = [1, 2, 4]
        
        for num_threads in thread_counts:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for _ in range(num_threads):
                    future = executor.submit(worker)
                    futures.append(future)
                
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            total_requests = sum(r["requests"] for r in results)
            overall_rps = total_requests / total_time
            
            print(f"Threads: {num_threads}, Total RPS: {overall_rps:.0f}, "
                  f"Total Time: {total_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_batch_vs_individual(self, is_odd_adapter):
        """Compare individual vs batch processing overhead."""
        numbers = list(range(100))
        
        # Test individual requests
        start_time = time.time()
        individual_results = []
        for number in numbers:
            result = await is_odd_adapter.predict({"number": number})
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Test batch processing (simulated)
        start_time = time.time()
        batch_tasks = []
        for number in numbers:
            batch_tasks.append(is_odd_adapter.predict({"number": number}))
        batch_results = await asyncio.gather(*batch_tasks)
        batch_time = time.time() - start_time
        
        # Verify results are the same
        assert len(individual_results) == len(batch_results) == 100
        for i in range(100):
            assert individual_results[i]["is_odd"] == batch_results[i]["is_odd"]
        
        individual_rps = 100 / individual_time
        batch_rps = 100 / batch_time
        speedup = batch_rps / individual_rps
        
        print(f"Individual processing: {individual_rps:.0f} RPS ({individual_time:.3f}s)")
        print(f"Batch processing: {batch_rps:.0f} RPS ({batch_time:.3f}s)")
        print(f"Speedup: {speedup:.1f}x")
        
        # Batch should be significantly faster
        assert batch_rps > individual_rps