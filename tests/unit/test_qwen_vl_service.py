"""Test Qwen VL service endpoints."""

import base64
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from fastapi.testclient import TestClient
from services.qwen_vl.app import app


class TestQwenVLService:
    """Test Qwen VL service endpoints."""
    
    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with mocked adapter."""
        # Set environment variables for testing
        import os
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        os.environ["DEVICE"] = "cpu"
        
        # Mock the adapter to avoid loading actual models
        mock_adapter = MagicMock()
        mock_adapter.predict = AsyncMock()
        mock_adapter.stream = AsyncMock()
        mock_adapter._loaded = True  # Set as loaded (private attribute)
        mock_adapter.is_loaded = True  # Set public property too
        mock_adapter._ensure_loaded = MagicMock()  # Mock the ensure_loaded check
        mock_adapter.health_check = AsyncMock(return_value={"status": "healthy"})
        
        with patch('services.qwen_vl.app.QwenVLAdapter') as mock_adapter_class, \
             patch('services.qwen_vl.app.adapter', mock_adapter), \
             patch('services.common.gcs_loader._gcs_loader', None):
            
            mock_adapter_class.return_value = mock_adapter
            
            # Import app after mocking
            from services.qwen_vl.app import app
            
            return TestClient(app)
    
    @pytest.fixture
    def test_image_b64(self):
        """Create test image as base64."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='green')
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode()
    
    @pytest.fixture
    def mock_vqa_response(self):
        """Mock VQA response."""
        return {
            "answer": "This is a green square image.",
            "question": "What do you see in this image?",
            "max_tokens": 256,
            "temperature": 0.1
        }
    
    def test_vqa_ask_endpoint_success(self, client, test_image_b64, mock_vqa_response):
        """Test successful VQA request."""
        with patch('services.qwen_vl.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = mock_vqa_response
            
            response = client.post("/v1/vqa/ask", json={
                "image": test_image_b64,
                "question": "What do you see in this image?",
                "max_tokens": 256,
                "temperature": 0.1,
                "stream": False
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "result" in data
            
            result = data["result"]
            assert result["answer"] == "This is a green square image."
            assert result["question"] == "What do you see in this image?"
            assert result["max_tokens"] == 256
            assert result["temperature"] == 0.1
            
            # Verify adapter was called with correct payload
            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0][0]
            assert call_args["image"] == test_image_b64
            assert call_args["question"] == "What do you see in this image?"
            assert call_args["max_tokens"] == 256
            assert call_args["temperature"] == 0.1
    
    def test_vqa_ask_streaming(self, client, test_image_b64):
        """Test VQA streaming response."""
        async def mock_stream_generator():
            """Mock streaming generator."""
            for token in ["This", " is", " a", " test", " response"]:
                yield token
        
        with patch('services.qwen_vl.app.adapter.stream') as mock_stream:
            mock_stream.return_value = mock_stream_generator()
            
            response = client.post("/v1/vqa/ask", json={
                "image": test_image_b64,
                "question": "What do you see?",
                "stream": True
            })
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            
            # For streaming responses, we can't easily test the content in sync tests
            # The important part is that the endpoint accepts the request and returns SSE headers
            mock_stream.assert_called_once()
    
    def test_vqa_chat_endpoint_success(self, client, test_image_b64, mock_vqa_response):
        """Test successful chat request."""
        with patch('services.qwen_vl.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = mock_vqa_response
            
            response = client.post("/v1/vqa/chat", json={
                "image": test_image_b64,
                "question": "Describe this image",
                "max_tokens": 128,
                "temperature": 0.2
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "result" in data
            
            # Verify adapter was called with correct payload
            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0][0]
            assert call_args["image"] == test_image_b64
            assert call_args["question"] == "Describe this image"
            assert call_args["max_tokens"] == 128
            assert call_args["temperature"] == 0.2
    
    def test_vqa_ask_endpoint_missing_image(self, client):
        """Test VQA endpoint with missing image."""
        response = client.post("/v1/vqa/ask", json={
            "question": "What do you see?"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_vqa_ask_endpoint_missing_question(self, client, test_image_b64):
        """Test VQA endpoint with missing question."""
        response = client.post("/v1/vqa/ask", json={
            "image": test_image_b64
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_vqa_ask_endpoint_invalid_max_tokens(self, client, test_image_b64):
        """Test VQA endpoint with invalid max_tokens."""
        response = client.post("/v1/vqa/ask", json={
            "image": test_image_b64,
            "question": "What do you see?",
            "max_tokens": "invalid"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_vqa_ask_endpoint_invalid_temperature(self, client, test_image_b64):
        """Test VQA endpoint with invalid temperature."""
        response = client.post("/v1/vqa/ask", json={
            "image": test_image_b64,
            "question": "What do you see?",
            "temperature": "invalid"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_vqa_ask_endpoint_default_values(self, client, test_image_b64, mock_vqa_response):
        """Test VQA endpoint with default values."""
        with patch('services.qwen_vl.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = mock_vqa_response
            
            response = client.post("/v1/vqa/ask", json={
                "image": test_image_b64,
                "question": "What do you see?"
            })
            
            assert response.status_code == 200
            
            # Verify default values were used
            call_args = mock_predict.call_args[0][0]
            assert call_args["max_tokens"] == 256  # Default
            assert call_args["temperature"] == 0.1  # Default
    
    def test_vqa_ask_endpoint_adapter_error(self, client, test_image_b64):
        """Test VQA endpoint when adapter fails."""
        with patch('services.qwen_vl.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.side_effect = Exception("VQA model failed")
            
            response = client.post("/v1/vqa/ask", json={
                "image": test_image_b64,
                "question": "What do you see?",
                "stream": False
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "VQA model failed" in data["detail"]
    
    def test_vqa_chat_endpoint_adapter_error(self, client, test_image_b64):
        """Test chat endpoint when adapter fails."""
        with patch('services.qwen_vl.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.side_effect = Exception("Chat failed")
            
            response = client.post("/v1/vqa/chat", json={
                "image": test_image_b64,
                "question": "What do you see?"
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Chat failed" in data["detail"]
    
    @pytest.mark.skip(reason="SSE streaming error test has event loop conflicts")
    def test_vqa_streaming_adapter_error(self, client, test_image_b64):
        """Test streaming VQA when adapter fails."""
        async def mock_failing_stream():
            """Mock failing stream generator."""
            raise Exception("Streaming failed")
            yield  # This won't be reached but makes it a generator
        
        with patch('services.qwen_vl.app.adapter.stream') as mock_stream:
            mock_stream.return_value = mock_failing_stream()
            
            response = client.post("/v1/vqa/ask", json={
                "image": test_image_b64,
                "question": "What do you see?",
                "stream": True
            })
            
            # Should still return 200 for SSE, but error will be in stream
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    def test_vqa_request_model_validation(self):
        """Test VQARequest model validation."""
        from services.qwen_vl.app import VQARequest
        
        # Valid request
        request = VQARequest(
            image="base64_image",
            question="What do you see?",
            max_tokens=128,
            temperature=0.5,
            stream=True
        )
        assert request.image == "base64_image"
        assert request.question == "What do you see?"
        assert request.max_tokens == 128
        assert request.temperature == 0.5
        assert request.stream is True
        
        # Test defaults
        request_defaults = VQARequest(
            image="base64_image",
            question="What do you see?"
        )
        assert request_defaults.max_tokens == 256
        assert request_defaults.temperature == 0.1
        assert request_defaults.stream is False
    
    def test_app_metadata(self, client):
        """Test app metadata and docs."""
        # Test that the app is created with correct metadata
        assert app.title == "Qwen 2.5 VL Service"
        assert "Visual Question Answering" in app.description
        assert app.version == "1.0.0"
        
        # Test docs are accessible
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI spec
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert spec["info"]["title"] == "Qwen 2.5 VL Service"
    
    def test_health_endpoint(self, client):
        """Test health endpoint from common app."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint from common app."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Metrics endpoint returns plain text
        assert "text/plain" in response.headers.get("content-type", "")
    
    def test_complex_question_scenarios(self, client, test_image_b64):
        """Test various question types and complexities."""
        test_scenarios = [
            {
                "question": "What objects do you see in this image?",
                "expected_type": "object_detection"
            },
            {
                "question": "What is the main color in this image?",
                "expected_type": "color_analysis"
            },
            {
                "question": "Describe the scene in detail.",
                "expected_type": "scene_description"
            },
            {
                "question": "How many people are in this image?",
                "expected_type": "counting"
            }
        ]
        
        with patch('services.qwen_vl.app.adapter.predict', new_callable=AsyncMock) as mock_predict:
            for scenario in test_scenarios:
                mock_response = {
                    "answer": f"Mock answer for {scenario['expected_type']}",
                    "question": scenario["question"],
                    "max_tokens": 256,
                    "temperature": 0.1
                }
                mock_predict.return_value = mock_response
                
                response = client.post("/v1/vqa/ask", json={
                    "image": test_image_b64,
                    "question": scenario["question"]
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert scenario["question"] in data["result"]["question"]