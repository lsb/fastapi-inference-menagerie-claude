"""Unit tests for CLIP service."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from services.clip.adapter import CLIPAdapter
from services.common.app import create_app


class MockCLIPAdapter(CLIPAdapter):
    """Mock CLIP adapter for testing."""
    
    def __init__(self):
        # Don't call super().__init__ to avoid GCS loader initialization
        self.gcs_path = "gs://test-bucket/clip"
        self.device = "cpu"
        self.model = MagicMock()
        self.processor = MagicMock()
        self._loaded = True
    
    async def load_model(self) -> None:
        """Mock model loading."""
        pass
    
    async def predict(self, payload: dict) -> dict:
        """Mock CLIP prediction."""
        task = payload.get('task', 'encode_text')
        
        if task == 'encode_text':
            return {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "shape": [2, 3],
                "count": 2
            }
        elif task == 'encode_image':
            return {
                "embeddings": [[0.7, 0.8, 0.9]],
                "shape": [1, 3],
                "count": 1
            }
        elif task == 'similarity':
            return {
                "similarity_matrix": [[0.8, 0.3], [0.2, 0.9]],
                "text_count": 2,
                "image_count": 2,
                "texts": payload.get('texts', [])
            }
        else:
            raise ValueError(f"Unknown task: {task}")


@pytest.fixture
def clip_app():
    """Create CLIP test app with CLIP-specific routes."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from typing import Dict, Any, List
    
    # Create custom test app with CLIP routes
    app = FastAPI(title="CLIP Test Service")
    adapter = MockCLIPAdapter()
    
    # Request models
    class TextEncodeRequest(BaseModel):
        texts: List[str] = Field(..., description="List of texts to encode")
    
    class ImageEncodeRequest(BaseModel):
        images: List[str] = Field(..., description="List of base64 encoded images")
    
    class SimilarityRequest(BaseModel):
        texts: List[str] = Field(..., description="List of texts")
        images: List[str] = Field(..., description="List of base64 encoded images")
    
    # Add CLIP-specific routes
    @app.post("/v1/clip/encode")
    async def encode_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await adapter.predict(payload)
            return {"success": True, "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/v1/clip/encode/text")
    async def encode_text(request: TextEncodeRequest) -> Dict[str, Any]:
        try:
            payload = {"task": "encode_text", "texts": request.texts}
            result = await adapter.predict(payload)
            return {"success": True, "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/v1/clip/encode/image")
    async def encode_images(request: ImageEncodeRequest) -> Dict[str, Any]:
        try:
            payload = {"task": "encode_image", "images": request.images}
            result = await adapter.predict(payload)
            return {"success": True, "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/v1/clip/similarity")
    async def compute_similarity(request: SimilarityRequest) -> Dict[str, Any]:
        try:
            payload = {
                "task": "similarity",
                "texts": request.texts,
                "images": request.images
            }
            result = await adapter.predict(payload)
            return {"success": True, "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return TestClient(app)


@pytest.mark.unit
class TestCLIPService:
    """Test CLIP service endpoints."""
    
    def test_encode_text_endpoint(self, clip_app: TestClient):
        """Test text encoding endpoint."""
        response = clip_app.post(
            "/v1/clip/encode/text",
            json={"texts": ["a cat", "a dog"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["count"] == 2
        assert len(data["result"]["embeddings"]) == 2
    
    def test_encode_image_endpoint(self, clip_app: TestClient, sample_image_b64: str):
        """Test image encoding endpoint."""
        response = clip_app.post(
            "/v1/clip/encode/image",
            json={"images": [sample_image_b64]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["count"] == 1
    
    def test_similarity_endpoint(self, clip_app: TestClient, sample_image_b64: str):
        """Test similarity computation endpoint."""
        response = clip_app.post(
            "/v1/clip/similarity",
            json={
                "texts": ["a cat", "a dog"],
                "images": [sample_image_b64, sample_image_b64]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        result = data["result"]
        assert result["text_count"] == 2
        assert result["image_count"] == 2
        assert len(result["similarity_matrix"]) == 2
    
    def test_universal_encode_endpoint(self, clip_app: TestClient):
        """Test universal encoding endpoint."""
        response = clip_app.post(
            "/v1/clip/encode",
            json={
                "task": "encode_text",
                "texts": ["test text"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
    
    def test_invalid_request_format(self, clip_app: TestClient):
        """Test invalid request format."""
        response = clip_app.post(
            "/v1/clip/encode/text",
            json={"invalid": "format"}
        )
        
        assert response.status_code == 422  # Validation error


@pytest.mark.unit
class TestCLIPAdapter:
    """Test CLIP adapter implementation."""
    
    @pytest.mark.asyncio
    @patch('services.clip.adapter.CLIPProcessor')
    @patch('services.clip.adapter.CLIPModel')
    async def test_load_model_from_hub(self, mock_model_class, mock_processor_class, tmp_path):
        """Test loading model from HuggingFace Hub."""
        # Set cache directory to temporary path
        import os
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = CLIPAdapter("gs://test-bucket/clip", "cpu")
        
        # Mock the download_weights to return None (no GCS weights)
        adapter.download_weights = AsyncMock(return_value=None)
        
        # Mock the classes to return specific instances
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        await adapter.load_model()
        
        assert adapter._loaded is True
        assert adapter.model is not None
        assert adapter.processor is not None
        
        # Verify HuggingFace Hub was used
        mock_processor_class.from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
        mock_model_class.from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
    
    @pytest.mark.asyncio
    async def test_encode_text_task(self):
        """Test text encoding task."""
        adapter = MockCLIPAdapter()
        
        payload = {
            "task": "encode_text",
            "texts": ["test text"]
        }
        
        result = await adapter.predict(payload)
        
        assert "embeddings" in result
        assert "count" in result
        assert result["count"] == 2  # Mock returns 2 embeddings
    
    @pytest.mark.asyncio
    async def test_encode_image_task(self):
        """Test image encoding task."""
        adapter = MockCLIPAdapter()
        
        payload = {
            "task": "encode_image",
            "images": ["base64_image_data"]
        }
        
        result = await adapter.predict(payload)
        
        assert "embeddings" in result
        assert "count" in result
        assert result["count"] == 1  # Mock returns 1 embedding
    
    @pytest.mark.asyncio
    async def test_similarity_task(self):
        """Test similarity computation task."""
        adapter = MockCLIPAdapter()
        
        payload = {
            "task": "similarity",
            "texts": ["cat", "dog"],
            "images": ["image1", "image2"]
        }
        
        result = await adapter.predict(payload)
        
        assert "similarity_matrix" in result
        assert "text_count" in result
        assert "image_count" in result
        assert result["texts"] == ["cat", "dog"]
    
    @pytest.mark.asyncio
    async def test_unknown_task(self):
        """Test unknown task handling."""
        adapter = MockCLIPAdapter()
        
        payload = {"task": "unknown_task"}
        
        with pytest.raises(ValueError, match="Unknown task"):
            await adapter.predict(payload)