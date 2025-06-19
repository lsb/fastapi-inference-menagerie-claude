"""Unit tests for model adapters."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.common.adapter import ModelAdapter
from tests.conftest import MockModelAdapter


@pytest.mark.unit
class TestModelAdapter:
    """Test ModelAdapter base class."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = MockModelAdapter(
            gcs_path="gs://test-bucket/test-model",
            device="cuda:0"
        )
        
        assert adapter.gcs_path == "gs://test-bucket/test-model"
        assert adapter.device == "cuda:0"
        assert adapter.is_loaded is True
        assert adapter.model is not None
    
    @pytest.mark.asyncio
    async def test_predict_method(self):
        """Test predict method."""
        adapter = MockModelAdapter()
        
        payload = {"test": "data"}
        result = await adapter.predict(payload)
        
        assert result["result"] == "mock_prediction"
        assert result["input"] == payload
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check method."""
        adapter = MockModelAdapter()
        
        health = await adapter.health_check()
        
        assert health["status"] == "healthy"
        assert health["model_class"] == "MockModelAdapter"
        assert health["device"] == "cpu"
        assert health["gcs_path"] == "gs://test-bucket/test-model"
    
    @pytest.mark.asyncio
    async def test_stream_method(self):
        """Test stream method default implementation."""
        adapter = MockModelAdapter()
        
        payload = {"test": "data"}
        results = []
        
        async for chunk in adapter.stream(payload):
            results.append(chunk)
        
        # Default stream implementation should yield string of predict result
        assert len(results) == 1
        assert "mock_prediction" in results[0]
    
    @pytest.mark.asyncio
    @patch('services.common.adapter.get_gcs_loader')
    async def test_download_weights_success(self, mock_get_loader):
        """Test successful weight downloading."""
        # Mock GCS loader
        mock_loader = MagicMock()
        mock_loader.check_exists.return_value = True
        mock_loader.download_weights.return_value = "/tmp/test-weights"
        mock_get_loader.return_value = mock_loader
        
        adapter = MockModelAdapter()
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = "/tmp/test-weights"
            mock_loop.return_value.run_in_executor = mock_executor
            
            weights_path = await adapter.download_weights()
            
            assert weights_path == "/tmp/test-weights"
            mock_loader.check_exists.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('services.common.adapter.get_gcs_loader')
    async def test_download_weights_not_exists(self, mock_get_loader):
        """Test weight downloading when GCS path doesn't exist."""
        # Mock GCS loader
        mock_loader = MagicMock()
        mock_loader.check_exists.return_value = False
        mock_get_loader.return_value = mock_loader
        
        adapter = MockModelAdapter()
        
        weights_path = await adapter.download_weights()
        
        assert weights_path is None
        mock_loader.check_exists.assert_called_once()
    
    def test_ensure_loaded_when_loaded(self):
        """Test _ensure_loaded when model is loaded."""
        adapter = MockModelAdapter()
        adapter._loaded = True
        
        # Should not raise exception
        adapter._ensure_loaded()
    
    def test_ensure_loaded_when_not_loaded(self):
        """Test _ensure_loaded when model is not loaded."""
        adapter = MockModelAdapter()
        adapter._loaded = False
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter._ensure_loaded()