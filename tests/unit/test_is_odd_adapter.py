"""Tests for is-odd adapter."""

import os
import pytest

from services.is_odd.adapter import IsOddAdapter


@pytest.mark.unit
class TestIsOddAdapter:
    """Test is-odd adapter functionality."""
    
    @pytest.fixture
    async def is_odd_adapter(self, tmp_path):
        """Create is-odd adapter."""
        # Set cache directory to temporary path
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        # Reset GCS loader singleton to pick up new cache dir
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = IsOddAdapter()
        await adapter.load_model()
        return adapter
    
    @pytest.mark.asyncio
    async def test_load_model(self, tmp_path):
        """Test model loading."""
        # Set cache directory to temporary path
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        # Reset GCS loader singleton to pick up new cache dir
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = IsOddAdapter()
        assert not adapter._loaded
        
        await adapter.load_model()
        assert adapter._loaded
    
    @pytest.mark.asyncio
    async def test_predict_odd_numbers(self, is_odd_adapter):
        """Test prediction for odd numbers."""
        # Test odd integers
        result = await is_odd_adapter.predict({"number": 1})
        assert result["is_odd"] is True
        assert result["number"] == 1
        
        result = await is_odd_adapter.predict({"number": 7})
        assert result["is_odd"] is True
        assert result["number"] == 7
        
        result = await is_odd_adapter.predict({"number": -3})
        assert result["is_odd"] is True
        assert result["number"] == -3
    
    @pytest.mark.asyncio
    async def test_predict_even_numbers(self, is_odd_adapter):
        """Test prediction for even numbers."""
        # Test even integers
        result = await is_odd_adapter.predict({"number": 0})
        assert result["is_odd"] is False
        assert result["number"] == 0
        
        result = await is_odd_adapter.predict({"number": 2})
        assert result["is_odd"] is False
        assert result["number"] == 2
        
        result = await is_odd_adapter.predict({"number": -4})
        assert result["is_odd"] is False
        assert result["number"] == -4
    
    @pytest.mark.asyncio
    async def test_predict_float_numbers(self, is_odd_adapter):
        """Test prediction for float numbers (converted to int)."""
        result = await is_odd_adapter.predict({"number": 3.7})
        assert result["is_odd"] is True
        assert result["number"] == 3
        
        result = await is_odd_adapter.predict({"number": 4.2})
        assert result["is_odd"] is False
        assert result["number"] == 4
        
        result = await is_odd_adapter.predict({"number": -1.9})
        assert result["is_odd"] is True
        assert result["number"] == -1
    
    @pytest.mark.asyncio
    async def test_predict_missing_number(self, is_odd_adapter):
        """Test prediction with missing number field."""
        with pytest.raises(ValueError, match="Missing 'number' field"):
            await is_odd_adapter.predict({})
    
    @pytest.mark.asyncio
    async def test_predict_invalid_type(self, is_odd_adapter):
        """Test prediction with invalid number type."""
        with pytest.raises(ValueError, match="Number must be int or float"):
            await is_odd_adapter.predict({"number": "not_a_number"})
    
    @pytest.mark.asyncio
    async def test_predict_without_loaded_model(self, tmp_path):
        """Test prediction without loading model first."""
        # Set cache directory to temporary path
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        # Reset GCS loader singleton to pick up new cache dir
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        adapter = IsOddAdapter()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await adapter.predict({"number": 5})
    
    @pytest.mark.asyncio
    async def test_result_format(self, is_odd_adapter):
        """Test that result contains expected fields."""
        result = await is_odd_adapter.predict({"number": 5})
        
        assert "is_odd" in result
        assert "number" in result
        assert "computation" in result
        assert isinstance(result["is_odd"], bool)
        assert isinstance(result["number"], int)
        assert isinstance(result["computation"], str)
    
    @pytest.mark.asyncio
    async def test_large_numbers(self, is_odd_adapter):
        """Test with large numbers."""
        result = await is_odd_adapter.predict({"number": 1000001})
        assert result["is_odd"] is True
        
        result = await is_odd_adapter.predict({"number": 1000000})
        assert result["is_odd"] is False