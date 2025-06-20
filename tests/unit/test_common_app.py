"""Unit tests for common FastAPI app."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_health_endpoint(test_app: TestClient):
    """Test health check endpoint."""
    response = test_app.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_class" in data
    assert "device" in data


@pytest.mark.unit
def test_root_endpoint(test_app: TestClient):
    """Test root endpoint."""
    response = test_app.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "Test Model Service" in data["message"]


@pytest.mark.unit
def test_metrics_endpoint(test_app: TestClient):
    """Test Prometheus metrics endpoint."""
    response = test_app.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    
    # Check for basic Prometheus metrics
    content = response.text
    assert "# HELP" in content
    assert "# TYPE" in content


@pytest.mark.unit
def test_cors_headers(test_app: TestClient):
    """Test CORS headers are present."""
    # Make a GET request with an Origin header to trigger CORS
    response = test_app.get("/health", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    
    # CORS headers should be present in response
    headers = response.headers
    assert "access-control-allow-origin" in headers
    assert headers["access-control-allow-origin"] == "*"