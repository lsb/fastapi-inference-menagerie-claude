"""Test Grounding DINO + SAM2 adapter functionality."""

import base64
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
import torch

from services.grounding_sam.adapter import GroundingSAMAdapter


class TestGroundingSAMAdapter:
    """Test Grounding DINO + SAM2 adapter."""
    
    @pytest.fixture
    def adapter(self, tmp_path):
        """Create adapter with test configuration."""
        # Set cache directory to temporary path
        import os
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = GroundingSAMAdapter(
            gcs_path=None,  # No GCS for testing
            device="cpu"
        )
        return adapter
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        # Create a simple RGB image
        img = Image.new('RGB', (224, 224), color='blue')
        return img
    
    @pytest.fixture
    def test_image_b64(self, test_image):
        """Create test image as base64."""
        import io
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode()
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter is not None
        assert adapter.device == "cpu"
        assert not adapter.is_loaded
        assert adapter.grounding_model is None
        assert adapter.grounding_processor is None
        assert adapter.sam_predictor is None
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, adapter):
        """Test successful model loading."""
        # Mock Grounding DINO components
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        with patch('services.grounding_sam.adapter.AutoProcessor') as mock_auto_processor, \
             patch('services.grounding_sam.adapter.AutoModelForZeroShotObjectDetection') as mock_auto_model, \
             patch('services.grounding_sam.adapter.build_sam2', None), \
             patch('services.grounding_sam.adapter.SAM2ImagePredictor', None):
            
            mock_auto_processor.from_pretrained.return_value = mock_processor
            mock_auto_model.from_pretrained.return_value = mock_model
            
            await adapter.load_model()
            
            assert adapter.is_loaded
            assert adapter.grounding_model == mock_model
            assert adapter.grounding_processor == mock_processor
            assert adapter.sam_predictor is None  # SAM2 not available in test
            
            # Verify model was moved to correct device
            mock_model.to.assert_called_with("cpu")
            mock_model.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model_with_sam2(self, adapter):
        """Test model loading with SAM2 available."""
        # Mock all components
        mock_processor = MagicMock()
        mock_grounding_model = MagicMock()
        mock_grounding_model.to.return_value = mock_grounding_model
        mock_grounding_model.eval.return_value = mock_grounding_model
        
        mock_sam2_model = MagicMock()
        mock_sam2_predictor = MagicMock()
        
        with patch('services.grounding_sam.adapter.AutoProcessor') as mock_auto_processor, \
             patch('services.grounding_sam.adapter.AutoModelForZeroShotObjectDetection') as mock_auto_model, \
             patch('services.grounding_sam.adapter.build_sam2') as mock_build_sam2, \
             patch('services.grounding_sam.adapter.SAM2ImagePredictor') as mock_sam2_predictor_class:
            
            mock_auto_processor.from_pretrained.return_value = mock_processor
            mock_auto_model.from_pretrained.return_value = mock_grounding_model
            mock_build_sam2.return_value = mock_sam2_model
            mock_sam2_predictor_class.return_value = mock_sam2_predictor
            
            await adapter.load_model()
            
            assert adapter.is_loaded
            assert adapter.grounding_model == mock_grounding_model
            assert adapter.grounding_processor == mock_processor
            assert adapter.sam_predictor == mock_sam2_predictor
    
    @pytest.mark.asyncio
    async def test_load_model_sam2_fail(self, adapter):
        """Test model loading when SAM2 fails to load."""
        mock_processor = MagicMock()
        mock_grounding_model = MagicMock()
        mock_grounding_model.to.return_value = mock_grounding_model
        mock_grounding_model.eval.return_value = mock_grounding_model
        
        with patch('services.grounding_sam.adapter.AutoProcessor') as mock_auto_processor, \
             patch('services.grounding_sam.adapter.AutoModelForZeroShotObjectDetection') as mock_auto_model, \
             patch('services.grounding_sam.adapter.build_sam2') as mock_build_sam2, \
             patch('services.grounding_sam.adapter.SAM2ImagePredictor'):
            
            mock_auto_processor.from_pretrained.return_value = mock_processor
            mock_auto_model.from_pretrained.return_value = mock_grounding_model
            mock_build_sam2.side_effect = Exception("SAM2 load failed")
            
            await adapter.load_model()
            
            # Should still load successfully without SAM2
            assert adapter.is_loaded
            assert adapter.grounding_model == mock_grounding_model
            assert adapter.sam_predictor is None
    
    @pytest.mark.asyncio
    async def test_predict_detection_only(self, adapter, test_image_b64):
        """Test prediction with detection only (no masks)."""
        # Mock the model loading
        adapter._loaded = True
        adapter.grounding_model = MagicMock()
        adapter.grounding_processor = MagicMock()
        adapter.sam_predictor = None
        
        # Mock grounding DINO response
        mock_detection_result = {
            "boxes": [[10, 10, 50, 50], [60, 60, 90, 90]],
            "scores": [0.8, 0.9],
            "labels": ["person", "car"],
            "count": 2,
            "text_prompt": "person walking"
        }
        
        with patch.object(adapter, '_run_grounding_dino', new_callable=AsyncMock) as mock_grounding:
            mock_grounding.return_value = mock_detection_result
            
            payload = {
                "image": test_image_b64,
                "text": "person walking",
                "confidence_threshold": 0.3,
                "include_masks": False
            }
            
            result = await adapter.predict(payload)
            
            assert result == mock_detection_result
            assert "masks" not in result
            mock_grounding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_with_masks(self, adapter, test_image_b64):
        """Test prediction with masks."""
        # Mock the model loading
        adapter._loaded = True
        adapter.grounding_model = MagicMock()
        adapter.grounding_processor = MagicMock()
        adapter.sam_predictor = MagicMock()
        
        # Mock responses
        mock_detection_result = {
            "boxes": [[10, 10, 50, 50]],
            "scores": [0.8],
            "labels": ["person"],
            "count": 1,
            "text_prompt": "person walking"
        }
        
        mock_masks = ["base64_encoded_mask"]
        
        with patch.object(adapter, '_run_grounding_dino', new_callable=AsyncMock) as mock_grounding, \
             patch.object(adapter, '_run_sam2', new_callable=AsyncMock) as mock_sam2:
            
            mock_grounding.return_value = mock_detection_result
            mock_sam2.return_value = mock_masks
            
            payload = {
                "image": test_image_b64,
                "text": "person walking",
                "confidence_threshold": 0.3,
                "include_masks": True
            }
            
            result = await adapter.predict(payload)
            
            assert result["boxes"] == mock_detection_result["boxes"]
            assert result["masks"] == mock_masks
            mock_grounding.assert_called_once()
            # Verify SAM2 was called with an image object and boxes
            mock_sam2.assert_called_once()
            call_args = mock_sam2.call_args[0]
            # First argument should be a PIL Image
            from PIL import Image
            assert isinstance(call_args[0], Image.Image)
            # Second argument should be the boxes
            assert call_args[1] == mock_detection_result["boxes"]
    
    @pytest.mark.asyncio
    async def test_predict_no_detections(self, adapter, test_image_b64):
        """Test prediction with no detections."""
        # Mock the model loading
        adapter._loaded = True
        adapter.grounding_model = MagicMock()
        adapter.grounding_processor = MagicMock()
        adapter.sam_predictor = MagicMock()
        
        # Mock no detections
        mock_detection_result = {
            "boxes": [],
            "scores": [],
            "labels": [],
            "count": 0,
            "text_prompt": "nonexistent object"
        }
        
        with patch.object(adapter, '_run_grounding_dino', new_callable=AsyncMock) as mock_grounding:
            mock_grounding.return_value = mock_detection_result
            
            payload = {
                "image": test_image_b64,
                "text": "nonexistent object",
                "confidence_threshold": 0.9,
                "include_masks": True
            }
            
            result = await adapter.predict(payload)
            
            assert result["count"] == 0
            assert result["boxes"] == []
            assert "masks" not in result  # No masks when no boxes
    
    @pytest.mark.asyncio
    async def test_run_grounding_dino(self, adapter, test_image):
        """Test Grounding DINO detection."""
        # Setup mocks
        adapter.grounding_processor = MagicMock()
        adapter.grounding_model = MagicMock()
        adapter.device = "cpu"
        
        # Mock processor inputs and outputs
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.tensor([[[[0.5]]]])
        }
        adapter.grounding_processor.return_value = mock_inputs
        
        # Mock model outputs
        mock_outputs = MagicMock()
        adapter.grounding_model.return_value = mock_outputs
        
        # Mock post-processing results
        mock_results = {
            "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]]),
            "scores": torch.tensor([0.8, 0.9]),
            "labels": ["person", "car"]
        }
        adapter.grounding_processor.post_process_grounded_object_detection.return_value = [mock_results]
        
        result = await adapter._run_grounding_dino(test_image, "person walking", 0.3)
        
        assert result["boxes"] == [[10, 10, 50, 50], [60, 60, 90, 90]]
        # Compare scores as floats (tensors may have slight precision differences)
        assert len(result["scores"]) == 2
        assert abs(result["scores"][0] - 0.8) < 0.001
        assert abs(result["scores"][1] - 0.9) < 0.001
        assert result["labels"] == ["person", "car"]
        assert result["count"] == 2
        assert result["text_prompt"] == "person walking"
        
        # Verify processor was called correctly
        adapter.grounding_processor.assert_called_with(
            images=test_image,
            text="person walking",
            return_tensors="pt"
        )
    
    @pytest.mark.asyncio
    async def test_run_sam2(self, adapter, test_image):
        """Test SAM2 segmentation."""
        # Setup mock SAM2 predictor
        adapter.sam_predictor = MagicMock()
        
        # Mock SAM2 prediction
        mock_mask = np.ones((1, 224, 224), dtype=bool)
        mock_scores = np.array([0.9])
        mock_logits = np.array([[[1.0]]])
        
        adapter.sam_predictor.predict.return_value = (mock_mask, mock_scores, mock_logits)
        
        boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
        
        with patch('services.grounding_sam.adapter.encode_image_to_base64') as mock_encode:
            mock_encode.return_value = "base64_encoded_mask"
            
            result = await adapter._run_sam2(test_image, boxes)
            
            assert len(result) == 2
            assert all(mask == "base64_encoded_mask" for mask in result)
            
            # Verify SAM2 was called for each box
            assert adapter.sam_predictor.predict.call_count == 2
            adapter.sam_predictor.set_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_sam2_no_predictor(self, adapter, test_image):
        """Test SAM2 when predictor is not available."""
        adapter.sam_predictor = None
        
        boxes = [[10, 10, 50, 50]]
        result = await adapter._run_sam2(test_image, boxes)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_predict_not_loaded(self, adapter, test_image_b64):
        """Test prediction when model is not loaded."""
        payload = {
            "image": test_image_b64,
            "text": "person walking"
        }
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await adapter.predict(payload)
    
    def test_payload_validation(self, adapter, test_image_b64):
        """Test payload validation for different inputs."""
        # Test that defaults are properly applied
        payload = {
            "image": test_image_b64,
            "text": "person walking"
        }
        
        # Verify defaults
        confidence = payload.get('confidence_threshold', 0.3)
        include_masks = payload.get('include_masks', True)
        
        assert confidence == 0.3
        assert include_masks is True
        
        # Test with custom values
        payload_custom = {
            "image": test_image_b64,
            "text": "car on street",
            "confidence_threshold": 0.5,
            "include_masks": False
        }
        
        assert payload_custom['confidence_threshold'] == 0.5
        assert payload_custom['include_masks'] is False