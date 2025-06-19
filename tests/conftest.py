"""Pytest configuration and fixtures."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from services.common.adapter import ModelAdapter
from services.common.app import create_app


class MockModelAdapter(ModelAdapter):
    """Mock model adapter for testing."""
    
    def __init__(self, gcs_path: str = "gs://test-bucket/test-model", device: str = "cpu"):
        super().__init__(gcs_path, device)
        self.model = MagicMock()
        self._loaded = True
    
    async def load_model(self) -> None:
        """Mock model loading."""
        self._loaded = True
    
    async def predict(self, payload: dict) -> dict:
        """Mock prediction."""
        return {"result": "mock_prediction", "input": payload}


@pytest.fixture
def mock_adapter() -> MockModelAdapter:
    """Create mock model adapter."""
    return MockModelAdapter()


@pytest.fixture
def test_app(mock_adapter: MockModelAdapter) -> TestClient:
    """Create test FastAPI app."""
    app = create_app(
        model_adapter=mock_adapter,
        title="Test Model Service",
        description="Test service for unit tests",
        version="0.1.0"
    )
    return TestClient(app)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_image_b64() -> str:
    """Sample base64 encoded image for testing."""
    # 1x1 red pixel PNG
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA0F1g1wAAAABJRU5ErkJggg=="


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing."""
    return ["a cat", "a dog", "a bird"]


# Async fixture helpers
@pytest.fixture
async def async_mock_adapter() -> AsyncGenerator[MockModelAdapter, None]:
    """Create async mock adapter."""
    adapter = MockModelAdapter()
    await adapter.load_model()
    yield adapter