"""Test Grounding DINO + SAM2 service endpoints."""

import base64
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import AsyncMock, patch, MagicMock
import os

from fastapi.testclient import TestClient


class TestGroundingSAMService:
    """Test Grounding DINO + SAM2 service endpoints."""
    
    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with mocked app."""
        # Set environment variables for testing
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        os.environ["DEVICE"] = "cpu"
        
        # Mock the adapter to avoid loading actual models
        mock_adapter = MagicMock()
        mock_adapter.predict = AsyncMock()
        mock_adapter._loaded = True  # Set as loaded (private attribute)
        mock_adapter.is_loaded = True  # Set public property too
        mock_adapter._ensure_loaded = MagicMock()  # Mock the ensure_loaded check
        mock_adapter.health_check = AsyncMock(return_value={"status": "healthy"})
        
        with patch('services.grounding_sam.app.GroundingSAMAdapter') as mock_adapter_class, \
             patch('services.grounding_sam.app.adapter', mock_adapter), \
             patch('services.common.gcs_loader._gcs_loader', None):
            
            mock_adapter_class.return_value = mock_adapter
            
            # Import app after mocking
            from services.grounding_sam.app import app
            
            return TestClient(app)
    
    @pytest.fixture
    def test_image_b64(self):
        """Create test image as base64."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode()
    
    @pytest.fixture
    def mock_adapter_response(self):
        """Mock adapter response."""
        return {
            "boxes": [[10, 10, 50, 50], [60, 60, 90, 90]],
            "scores": [0.8, 0.9],
            "labels": ["person", "car"],
            "count": 2,
            "text_prompt": "person walking",
            "masks": ["base64_mask_1", "base64_mask_2"]
        }
    
    def test_segment_endpoint_success(self, client, test_image_b64, mock_adapter_response):
        """Test successful object segmentation."""
        with patch('services.grounding_sam.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = mock_adapter_response
            
            response = client.post("/v1/ground/segment", json={
                "image": test_image_b64,
                "text": "person walking",
                "confidence_threshold": 0.3,
                "include_masks": True
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "result" in data
            
            result = data["result"]
            assert "boxes" in result
            assert "scores" in result
            assert "labels" in result
            assert "masks" in result
            assert result["count"] == 2
            
            # Verify adapter was called with correct payload
            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0][0]
            assert call_args["image"] == test_image_b64
            assert call_args["text"] == "person walking"
            assert call_args["confidence_threshold"] == 0.3
            assert call_args["include_masks"] is True
    
    def test_detect_endpoint_success(self, client, test_image_b64, mock_adapter_response):
        """Test successful object detection (boxes only)."""
        # Remove masks from response for detection-only
        detection_response = {k: v for k, v in mock_adapter_response.items() if k != "masks"}
        
        with patch('services.grounding_sam.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = detection_response
            
            response = client.post("/v1/ground/detect", json={
                "image": test_image_b64,
                "text": "person walking",
                "confidence_threshold": 0.5
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "result" in data
            
            result = data["result"]
            assert "boxes" in result
            assert "scores" in result
            assert "labels" in result
            assert "masks" not in result  # Should not include masks for detection
            
            # Verify adapter was called with include_masks=False
            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0][0]
            assert call_args["include_masks"] is False
    
    def test_segment_endpoint_missing_image(self, client):
        """Test segment endpoint with missing image."""
        response = client.post("/v1/ground/segment", json={
            "text": "person walking"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_segment_endpoint_missing_text(self, client, test_image_b64):
        """Test segment endpoint with missing text."""
        response = client.post("/v1/ground/segment", json={
            "image": test_image_b64
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_segment_endpoint_invalid_confidence_threshold(self, client, test_image_b64):
        """Test segment endpoint with invalid confidence threshold."""
        response = client.post("/v1/ground/segment", json={
            "image": test_image_b64,
            "text": "person walking",
            "confidence_threshold": "invalid"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_segment_endpoint_default_values(self, client, test_image_b64, mock_adapter_response):
        """Test segment endpoint with default values."""
        with patch('services.grounding_sam.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = mock_adapter_response
            
            response = client.post("/v1/ground/segment", json={
                "image": test_image_b64,
                "text": "person walking"
            })
            
            assert response.status_code == 200
            
            # Verify default values were used
            call_args = mock_predict.call_args[0][0]
            assert call_args["confidence_threshold"] == 0.3  # Default
            assert call_args["include_masks"] is True  # Default
    
    def test_segment_endpoint_adapter_error(self, client, test_image_b64):
        """Test segment endpoint when adapter fails."""
        with patch('services.grounding_sam.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.side_effect = Exception("Model failed")
            
            response = client.post("/v1/ground/segment", json={
                "image": test_image_b64,
                "text": "person walking"
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Model failed" in data["detail"]
    
    def test_detect_endpoint_adapter_error(self, client, test_image_b64):
        """Test detect endpoint when adapter fails."""
        with patch('services.grounding_sam.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.side_effect = Exception("Detection failed")
            
            response = client.post("/v1/ground/detect", json={
                "image": test_image_b64,
                "text": "person walking"
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Detection failed" in data["detail"]
    
    def test_segment_request_model_validation(self):
        """Test SegmentRequest model validation."""
        from services.grounding_sam.app import SegmentRequest
        
        # Valid request
        request = SegmentRequest(
            image="base64_image",
            text="person walking",
            confidence_threshold=0.5,
            include_masks=False
        )
        assert request.image == "base64_image"
        assert request.text == "person walking"
        assert request.confidence_threshold == 0.5
        assert request.include_masks is False
        
        # Test defaults
        request_defaults = SegmentRequest(
            image="base64_image",
            text="person walking"
        )
        assert request_defaults.confidence_threshold == 0.3
        assert request_defaults.include_masks is True
    
    def test_app_metadata(self, client):
        """Test app metadata and docs."""
        # Test that the app is created with correct metadata
        assert client.app.title == "Grounding DINO + SAM2 Service"
        assert "Object detection and segmentation" in client.app.description
        assert client.app.version == "1.0.0"
        
        # Test docs are accessible
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI spec
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert spec["info"]["title"] == "Grounding DINO + SAM2 Service"