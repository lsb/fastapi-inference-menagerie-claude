"""HTTP performance tests for is-odd service with actual network calls."""

import pytest
import time
import asyncio
import httpx
import threading
import uvicorn
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.is_odd.adapter import IsOddAdapter
from services.common.app import create_app


@pytest.mark.performance
class TestHTTPIsOddPerformance:
    """Test is-odd service with actual HTTP calls over the network."""
    
    @pytest.fixture(scope="class")
    async def is_odd_service_server(self):
        """Start is-odd service on a test port with real HTTP server."""
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/is_odd_http_perf_cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        # Create is-odd adapter
        adapter = IsOddAdapter()
        await adapter.load_model()
        
        # Create FastAPI app with routes
        app = create_app(adapter)
        
        # Add is-odd specific routes
        class IsOddRequest(BaseModel):
            number: float = Field(..., description="Number to check if odd")
        
        class IsOddResponse(BaseModel):
            success: bool = Field(..., description="Whether the request succeeded")
            result: dict = Field(..., description="Prediction result")
        
        @app.post("/v1/is-odd/predict", response_model=IsOddResponse)
        async def predict_is_odd(request: IsOddRequest) -> IsOddResponse:
            try:
                result = await adapter.predict({"number": request.number})
                return IsOddResponse(success=True, result=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Start server in background thread
        test_port = 8902
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
                
                # Signal that server is starting
                server_ready.set()
                
                # Run server (this blocks)
                asyncio.run(server.serve())
            except Exception as e:
                server_exception = e
                server_ready.set()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        server_ready.wait(timeout=10)
        
        if server_exception:
            raise server_exception
        
        # Give server a moment to fully start
        await asyncio.sleep(1)
        
        base_url = f"http://127.0.0.1:{test_port}"
        
        # Test server is responding
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{base_url}/health")
                assert response.status_code == 200
            except Exception as e:
                pytest.skip(f"Server failed to start: {e}")
        
        yield base_url
        
        # Cleanup happens when test ends (daemon thread will die)
    
    @pytest.mark.asyncio
    async def test_http_is_odd_latency(self, is_odd_service_server):
        """Test HTTP latency for is-odd computation with network calls."""
        base_url = is_odd_service_server
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Warm up
            await client.post(f"{base_url}/v1/is-odd/predict", json={"number": 42})
            
            # Measure single is-odd computation over HTTP
            start_time = time.time()
            response = await client.post(f"{base_url}/v1/is-odd/predict", json={"number": 13})
            end_time = time.time()
            
            assert response.status_code == 200
            result = response.json()
            
            latency = end_time - start_time
            
            assert result["success"] == True
            assert result["result"]["is_odd"] == True
            assert result["result"]["number"] == 13
            
            print(f"HTTP Is-odd computation latency: {latency*1000:.3f}ms")
            
            # HTTP should still be very fast for is-odd
            assert latency < 1.0  # Should be under 1 second for HTTP
    
    @pytest.mark.asyncio
    async def test_http_is_odd_throughput(self, is_odd_service_server):
        """Test HTTP throughput for is-odd with real network calls."""
        base_url = is_odd_service_server
        num_requests = 10000  # Test with 10K requests over HTTP
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Run individual HTTP requests to get latency distribution
            latencies = []
            start_time = time.time()
            
            for i in range(num_requests):
                number = i + 1  # Start from 1
                
                req_start = time.time()
                response = await client.post(f"{base_url}/v1/is-odd/predict", json={"number": number})
                req_end = time.time()
                
                assert response.status_code == 200
                result = response.json()
                
                latencies.append((req_end - req_start) * 1000)  # Convert to ms
                
                # Verify correctness
                expected_is_odd = bool(number % 2)
                assert result["success"] == True
                assert result["result"]["is_odd"] == expected_is_odd
                assert result["result"]["number"] == number
            
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
            
            print(f"\nHTTP Is-Odd Performance ({num_requests:,} requests):")
            print(f"  Total time:    {total_time:.3f}s")
            print(f"  Throughput:    {rps:,.0f} requests/second")
            print(f"  Latency stats:")
            print(f"    Average:     {avg_latency:.3f}ms")
            print(f"    Min:         {min_latency:.3f}ms")
            print(f"    Max:         {max_latency:.3f}ms")
            print(f"    P50:         {p50_latency:.3f}ms")
            print(f"    P95:         {p95_latency:.3f}ms")
            print(f"    P99:         {p99_latency:.3f}ms")
            
            # HTTP is-odd should still be reasonably fast
            assert rps > 100  # At least 100 RPS for HTTP is-odd
    
    @pytest.mark.asyncio
    async def test_http_vs_direct_is_odd_overhead(self, is_odd_service_server):
        """Compare HTTP vs direct adapter call overhead for is-odd."""
        base_url = is_odd_service_server
        num_iterations = 1000
        
        # Test direct adapter calls
        import os
        os.environ.setdefault("CACHE_DIR", "/tmp/is_odd_direct_comparison_cache")
        
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        direct_adapter = IsOddAdapter()
        await direct_adapter.load_model()
        
        # Direct calls
        direct_latencies = []
        start_time = time.time()
        for i in range(num_iterations):
            number = i + 1
            req_start = time.time()
            result = await direct_adapter.predict({"number": number})
            req_end = time.time()
            direct_latencies.append((req_end - req_start) * 1000)
            
            # Verify correctness
            expected_is_odd = bool(number % 2)
            assert result["is_odd"] == expected_is_odd
        direct_total_time = time.time() - start_time
        
        # HTTP calls
        http_latencies = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            for i in range(num_iterations):
                number = i + 1
                req_start = time.time()
                response = await client.post(f"{base_url}/v1/is-odd/predict", json={"number": number})
                req_end = time.time()
                http_latencies.append((req_end - req_start) * 1000)
                
                assert response.status_code == 200
                result = response.json()
                expected_is_odd = bool(number % 2)
                assert result["success"] == True
                assert result["result"]["is_odd"] == expected_is_odd
                assert result["result"]["number"] == number
            http_total_time = time.time() - start_time
        
        # Calculate averages and throughput
        direct_avg = sum(direct_latencies) / len(direct_latencies)
        http_avg = sum(http_latencies) / len(http_latencies)
        direct_rps = num_iterations / direct_total_time
        http_rps = num_iterations / http_total_time
        
        overhead = http_avg - direct_avg
        overhead_percentage = (overhead / direct_avg) * 100
        rps_ratio = direct_rps / http_rps
        
        print(f"\nHTTP vs Direct Is-Odd Overhead Comparison ({num_iterations:,} iterations):")
        print(f"  Direct adapter:")
        print(f"    Avg latency:     {direct_avg:.3f}ms")
        print(f"    Throughput:      {direct_rps:,.0f} RPS")
        print(f"  HTTP requests:")
        print(f"    Avg latency:     {http_avg:.3f}ms")
        print(f"    Throughput:      {http_rps:,.0f} RPS")
        print(f"  HTTP overhead:")
        print(f"    Latency increase: {overhead:.3f}ms ({overhead_percentage:.1f}%)")
        print(f"    Throughput ratio: {rps_ratio:.1f}x slower")
        
        # HTTP should have noticeable overhead even for simple operations
        assert http_avg > direct_avg, "HTTP should be slower than direct calls"
        assert overhead > 0.1, "HTTP overhead should be at least 0.1ms"
        assert http_rps < direct_rps, "HTTP throughput should be lower than direct calls"