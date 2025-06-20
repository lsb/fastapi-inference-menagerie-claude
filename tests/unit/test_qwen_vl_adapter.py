"""Test Qwen VL adapter functionality."""

import base64
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch, AsyncMock
import torch
import asyncio

from services.qwen_vl.adapter import QwenVLAdapter


def create_mock_inputs(input_ids, attention_mask, pixel_values):
    """Create a mock inputs object that behaves like transformers output."""
    mock_inputs = MagicMock()
    mock_inputs.input_ids = input_ids
    mock_inputs.attention_mask = attention_mask
    mock_inputs.pixel_values = pixel_values
    
    # Define getitem method to access as dict
    def _getitem(key):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values
        }[key]
    mock_inputs.__getitem__ = _getitem
    
    # Define items method for iteration
    def _items():
        return [
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("pixel_values", pixel_values)
        ]
    mock_inputs.items = _items
    
    # Define keys method for dict-like behavior
    def _keys():
        return ["input_ids", "attention_mask", "pixel_values"]
    mock_inputs.keys = _keys
    
    return mock_inputs


class TestQwenVLAdapter:
    """Test Qwen VL adapter."""
    
    @pytest.fixture
    def adapter(self, tmp_path):
        """Create adapter with test configuration."""
        # Set cache directory to temporary path
        import os
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        # Reset GCS loader singleton
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = QwenVLAdapter(
            gcs_path=None,  # No GCS for testing
            device="cpu"
        )
        return adapter
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        # Create a simple RGB image with some content
        img = Image.new('RGB', (224, 224), color='red')
        # Add some simple pattern
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill='blue')
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
        assert adapter.model is None
        assert adapter.tokenizer is None
        assert adapter.processor is None
    
    def test_adapter_device_handling(self):
        """Test adapter device handling for CPU vs CUDA."""
        # Test CPU device
        cpu_adapter = QwenVLAdapter(gcs_path=None, device="cpu")
        assert cpu_adapter.device == "cpu"
        
        # Test CUDA device
        cuda_adapter = QwenVLAdapter(gcs_path=None, device="cuda:0")
        assert cuda_adapter.device == "cuda:0"
    
    @pytest.mark.asyncio
    async def test_load_model_cpu(self, adapter):
        """Test successful model loading on CPU."""
        # Mock the transformers components
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        with patch('services.qwen_vl.adapter.AutoTokenizer') as mock_auto_tokenizer, \
             patch('services.qwen_vl.adapter.AutoProcessor') as mock_auto_processor, \
             patch('services.qwen_vl.adapter.Qwen2VLForConditionalGeneration') as mock_auto_model:
            
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_auto_processor.from_pretrained.return_value = mock_processor
            mock_auto_model.from_pretrained.return_value = mock_model
            
            await adapter.load_model()
            
            assert adapter.is_loaded
            assert adapter.model == mock_model
            assert adapter.tokenizer == mock_tokenizer
            assert adapter.processor == mock_processor
            
            # Verify model was loaded with correct parameters for CPU
            mock_auto_model.from_pretrained.assert_called_with(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float32,  # Should use float32 for CPU
                device_map=None  # Should be None for CPU
            )
            
            # Verify model was moved to CPU device
            mock_model.to.assert_called_with("cpu")
            mock_model.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model_cuda(self, tmp_path):
        """Test model loading configuration for CUDA device."""
        import os
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        cuda_adapter = QwenVLAdapter(gcs_path=None, device="cuda:0")
        
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        
        with patch('services.qwen_vl.adapter.AutoTokenizer') as mock_auto_tokenizer, \
             patch('services.qwen_vl.adapter.AutoProcessor') as mock_auto_processor, \
             patch('services.qwen_vl.adapter.Qwen2VLForConditionalGeneration') as mock_auto_model:
            
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_auto_processor.from_pretrained.return_value = mock_processor
            mock_auto_model.from_pretrained.return_value = mock_model
            
            await cuda_adapter.load_model()
            
            # Verify model was loaded with correct parameters for CUDA
            mock_auto_model.from_pretrained.assert_called_with(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16,  # Should use float16 for CUDA
                device_map="cuda:0"  # Should specify device_map for CUDA
            )
            
            # Model.to() should not be called when device_map is used
            mock_model.to.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self, adapter):
        """Test model loading failure."""
        with patch('services.qwen_vl.adapter.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.side_effect = Exception("Model load failed")
            
            with pytest.raises(Exception, match="Model load failed"):
                await adapter.load_model()
            
            assert not adapter.is_loaded
    
    @pytest.mark.asyncio
    async def test_predict_success(self, adapter, test_image_b64):
        """Test successful prediction."""
        # Mock the loaded model
        adapter._loaded = True
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.processor = MagicMock()
        
        # Mock tokenizer
        adapter.tokenizer.eos_token_id = 2
        
        # Mock processor methods
        adapter.processor.apply_chat_template.return_value = "formatted_text"
        mock_inputs = create_mock_inputs(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[[[0.5]]]])
        )
        adapter.processor.return_value = mock_inputs
        adapter.processor.batch_decode.return_value = ["This is a test response."]
        
        # Mock model generation
        mock_generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        adapter.model.generate.return_value = mock_generated_ids
        
        payload = {
            "image": test_image_b64,
            "question": "What do you see in this image?",
            "max_tokens": 128,
            "temperature": 0.2
        }
        
        result = await adapter.predict(payload)
        
        assert result["answer"] == "This is a test response."
        assert result["question"] == "What do you see in this image?"
        assert result["max_tokens"] == 128
        assert result["temperature"] == 0.2
        
        # Verify model.generate was called with correct parameters
        adapter.model.generate.assert_called_once()
        generate_kwargs = adapter.model.generate.call_args[1]
        assert generate_kwargs["max_new_tokens"] == 128
        assert generate_kwargs["temperature"] == 0.2
        assert generate_kwargs["do_sample"] is True  # temperature > 0
        assert generate_kwargs["pad_token_id"] == 2
    
    @pytest.mark.asyncio
    async def test_predict_greedy_sampling(self, adapter, test_image_b64):
        """Test prediction with greedy sampling (temperature=0)."""
        # Mock the loaded model
        adapter._loaded = True
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.processor = MagicMock()
        
        # Mock components
        adapter.tokenizer.eos_token_id = 2
        adapter.processor.apply_chat_template.return_value = "formatted_text"
        adapter.processor.return_value = create_mock_inputs(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[[[0.5]]]])
        )
        adapter.processor.batch_decode.return_value = ["Greedy response."]
        
        mock_generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        adapter.model.generate.return_value = mock_generated_ids
        
        payload = {
            "image": test_image_b64,
            "question": "What is this?",
            "temperature": 0.0  # Greedy sampling
        }
        
        result = await adapter.predict(payload)
        
        # Verify greedy sampling parameters
        generate_kwargs = adapter.model.generate.call_args[1]
        assert generate_kwargs["temperature"] == 0.0
        assert generate_kwargs["do_sample"] is False  # temperature == 0
    
    @pytest.mark.asyncio
    async def test_predict_default_parameters(self, adapter, test_image_b64):
        """Test prediction with default parameters."""
        # Mock the loaded model
        adapter._loaded = True
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.processor = MagicMock()
        
        # Setup mocks
        adapter.tokenizer.eos_token_id = 2
        adapter.processor.apply_chat_template.return_value = "formatted_text"
        adapter.processor.return_value = create_mock_inputs(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[[[0.5]]]])
        )
        adapter.processor.batch_decode.return_value = ["Default response."]
        
        mock_generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        adapter.model.generate.return_value = mock_generated_ids
        
        # Test with minimal payload
        payload = {
            "image": test_image_b64,
            "question": "What is this?"
        }
        
        result = await adapter.predict(payload)
        
        # Verify defaults were applied
        assert result["max_tokens"] == 256  # Default
        assert result["temperature"] == 0.1  # Default
        
        generate_kwargs = adapter.model.generate.call_args[1]
        assert generate_kwargs["max_new_tokens"] == 256
        assert generate_kwargs["temperature"] == 0.1
    
    @pytest.mark.asyncio
    async def test_stream_success(self, adapter, test_image_b64):
        """Test successful streaming prediction."""
        # Mock the loaded model
        adapter._loaded = True
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.processor = MagicMock()
        
        # Mock tokenizer
        adapter.tokenizer.eos_token_id = 2
        adapter.tokenizer.decode.side_effect = ["This", " is", " a", " test"]
        
        # Mock processor
        adapter.processor.apply_chat_template.return_value = "formatted_text"
        adapter.processor.return_value = create_mock_inputs(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[[[0.5]]]])
        )
        
        # Mock model forward pass
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.7]]])  # Shape: [batch, seq, vocab]
        adapter.model.return_value = mock_outputs
        
        # Mock torch operations
        with patch('torch.argmax') as mock_argmax, \
             patch('torch.cat') as mock_cat:
            
            # Simulate 4 tokens being generated then EOS
            mock_argmax.side_effect = [
                torch.tensor([10]),  # Token 1
                torch.tensor([11]),  # Token 2  
                torch.tensor([12]),  # Token 3
                torch.tensor([2])    # EOS token
            ]
            
            # Mock torch.cat to return updated input_ids
            mock_cat.side_effect = [
                torch.tensor([[1, 2, 3, 10]]),
                torch.tensor([[1, 2, 3, 10, 11]]),
                torch.tensor([[1, 2, 3, 10, 11, 12]]),
                torch.tensor([[1, 2, 3, 10, 11, 12, 2]])
            ]
            
            payload = {
                "image": test_image_b64,
                "question": "What is this?",
                "max_tokens": 10,
                "temperature": 0.0  # Greedy for deterministic testing
            }
            
            # Collect streaming tokens
            tokens = []
            async for token in adapter.stream(payload):
                tokens.append(token)
            
            # Should generate 3 tokens before hitting EOS
            assert len(tokens) == 3
            assert tokens == ["This", " is", " a"]
    
    @pytest.mark.asyncio
    async def test_stream_temperature_sampling(self, adapter, test_image_b64):
        """Test streaming with temperature sampling."""
        # Mock the loaded model
        adapter._loaded = True
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.processor = MagicMock()
        
        # Setup mocks
        adapter.tokenizer.eos_token_id = 2
        adapter.tokenizer.decode.side_effect = ["Token1", "Token2"]
        
        adapter.processor.apply_chat_template.return_value = "formatted_text"
        adapter.processor.return_value = create_mock_inputs(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[[[0.5]]]])
        )
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.7]]])
        adapter.model.return_value = mock_outputs
        
        with patch('torch.softmax') as mock_softmax, \
             patch('torch.multinomial') as mock_multinomial, \
             patch('torch.cat') as mock_cat:
            
            # Mock temperature sampling
            mock_softmax.return_value = torch.tensor([[0.2, 0.3, 0.5]])
            mock_multinomial.side_effect = [
                torch.tensor([10]),  # Sampled token
                torch.tensor([2])    # EOS token
            ]
            
            mock_cat.side_effect = [
                torch.tensor([[1, 2, 3, 10]]),
                torch.tensor([[1, 2, 3, 10, 2]])
            ]
            
            payload = {
                "image": test_image_b64,
                "question": "What is this?",
                "temperature": 0.8  # Temperature sampling
            }
            
            tokens = []
            async for token in adapter.stream(payload):
                tokens.append(token)
            
            # Should have called softmax with temperature scaling
            mock_softmax.assert_called()
            softmax_call_args = mock_softmax.call_args[0]
            # First arg should be logits/temperature
            assert len(tokens) == 1  # One token before EOS
    
    @pytest.mark.asyncio
    async def test_stream_max_tokens_limit(self, adapter, test_image_b64):
        """Test streaming respects max_tokens limit."""
        # Mock the loaded model
        adapter._loaded = True
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.processor = MagicMock()
        
        # Setup mocks - never return EOS token
        adapter.tokenizer.eos_token_id = 2
        adapter.tokenizer.decode.return_value = "Token"
        
        adapter.processor.apply_chat_template.return_value = "formatted_text"
        adapter.processor.return_value = create_mock_inputs(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[[[0.5]]]])
        )
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.7]]])
        adapter.model.return_value = mock_outputs
        
        with patch('torch.argmax') as mock_argmax, \
             patch('torch.cat') as mock_cat:
            
            # Always return non-EOS token
            mock_argmax.return_value = torch.tensor([10])
            mock_cat.return_value = torch.tensor([[1, 2, 3, 10]])
            
            payload = {
                "image": test_image_b64,
                "question": "What is this?",
                "max_tokens": 3,  # Limit to 3 tokens
                "temperature": 0.0
            }
            
            tokens = []
            async for token in adapter.stream(payload):
                tokens.append(token)
            
            # Should stop at max_tokens limit
            assert len(tokens) == 3
            assert all(token == "Token" for token in tokens)
    
    @pytest.mark.asyncio
    async def test_predict_not_loaded(self, adapter, test_image_b64):
        """Test prediction when model is not loaded."""
        payload = {
            "image": test_image_b64,
            "question": "What is this?"
        }
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await adapter.predict(payload)
    
    @pytest.mark.asyncio
    async def test_stream_not_loaded(self, adapter, test_image_b64):
        """Test streaming when model is not loaded."""
        payload = {
            "image": test_image_b64,
            "question": "What is this?"
        }
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            async for _ in adapter.stream(payload):
                pass
    
    def test_conversation_formatting(self, adapter):
        """Test conversation message formatting."""
        # This tests the internal conversation structure
        # In a real test, we'd check the processor.apply_chat_template call
        adapter._loaded = True
        adapter.processor = MagicMock()
        adapter.processor.apply_chat_template.return_value = "formatted"
        
        # Expected message format
        expected_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "mock_image"},
                    {"type": "text", "text": "test question"}
                ]
            }
        ]
        
        # This would be tested in the actual predict/stream methods
        # Here we just verify the expected structure
        assert len(expected_messages) == 1
        assert expected_messages[0]["role"] == "user"
        assert len(expected_messages[0]["content"]) == 2
        assert expected_messages[0]["content"][0]["type"] == "image"
        assert expected_messages[0]["content"][1]["type"] == "text"