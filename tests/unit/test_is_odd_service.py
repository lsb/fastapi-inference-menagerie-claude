"""Tests for is-odd service endpoints."""

import os
import pytest
from fastapi.testclient import TestClient

# Set cache directory before importing app
os.environ["CACHE_DIR"] = "/tmp/test_is_odd_cache"

from services.is_odd.app import app, adapter


@pytest.mark.unit
class TestIsOddService:
    """Test is-odd service endpoints."""
    
    @pytest.fixture
    async def client(self):
        """Create test client with loaded model."""
        # Manually load the model since TestClient doesn't always trigger lifespan
        if not adapter._loaded:
            await adapter.load_model()
        
        return TestClient(app)
    
    def test_predict_odd_number(self, client):
        """Test single odd number prediction."""
        response = client.post(
            "/v1/is-odd/predict",
            json={"number": 7}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["is_odd"] is True
        assert data["result"]["number"] == 7
    
    def test_predict_even_number(self, client):
        """Test single even number prediction."""
        response = client.post(
            "/v1/is-odd/predict",
            json={"number": 8}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["is_odd"] is False
        assert data["result"]["number"] == 8
    
    def test_predict_zero(self, client):
        """Test zero (even) prediction."""
        response = client.post(
            "/v1/is-odd/predict",
            json={"number": 0}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["is_odd"] is False
        assert data["result"]["number"] == 0
    
    def test_predict_negative_odd(self, client):
        """Test negative odd number."""
        response = client.post(
            "/v1/is-odd/predict",
            json={"number": -5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["is_odd"] is True
        assert data["result"]["number"] == -5
    
    def test_predict_float(self, client):
        """Test float number (truncated to int)."""
        response = client.post(
            "/v1/is-odd/predict",
            json={"number": 3.8}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["is_odd"] is True
        assert data["result"]["number"] == 3
    
    def test_predict_invalid_request(self, client):
        """Test invalid request format."""
        response = client.post(
            "/v1/is-odd/predict",
            json={"wrong_field": 5}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict(self, client):
        """Test batch prediction endpoint."""
        response = client.post(
            "/v1/is-odd/batch",
            json=[1, 2, 3, 4, 5]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["count"] == 5
        
        results = data["result"]["batch_results"]
        assert len(results) == 5
        
        # Check each result
        assert results[0]["is_odd"] is True   # 1
        assert results[1]["is_odd"] is False  # 2
        assert results[2]["is_odd"] is True   # 3
        assert results[3]["is_odd"] is False  # 4
        assert results[4]["is_odd"] is True   # 5
    
    def test_batch_predict_empty(self, client):
        """Test batch prediction with empty list."""
        response = client.post(
            "/v1/is-odd/batch",
            json=[]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["count"] == 0
        assert data["result"]["batch_results"] == []
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus metrics
        assert "model_requests_total" in response.text
    
    def test_openapi_docs(self, client):
        """Test OpenAPI documentation."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "Is-Odd Demo Service" in openapi_spec["info"]["title"]