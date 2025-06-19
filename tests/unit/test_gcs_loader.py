"""Unit tests for GCS loader."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from services.common.gcs_loader import GCSLoader, get_gcs_loader
from services.common.utils import validate_gcs_path, extract_bucket_and_path


@pytest.mark.unit
class TestGCSLoader:
    """Test GCS loader functionality."""
    
    def test_initialization(self, temp_dir: Path):
        """Test GCS loader initialization."""
        loader = GCSLoader(
            cache_dir=temp_dir,
            enable_cache=True,
            project="test-project"
        )
        
        assert loader.cache_dir == temp_dir
        assert loader.enable_cache is True
        assert loader.project == "test-project"
    
    def test_initialization_default_cache(self):
        """Test initialization with default cache directory."""
        loader = GCSLoader()
        
        assert loader.cache_dir == Path("/var/cache/zoo")
        assert loader.enable_cache is True
    
    @patch('services.common.gcs_loader.storage.Client')
    def test_gcs_client_property(self, mock_client_class):
        """Test GCS client property."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        loader = GCSLoader(project="test-project")
        client = loader.gcs_client
        
        assert client == mock_client
        mock_client_class.assert_called_once_with(project="test-project")
    
    @patch('services.common.gcs_loader.gcsfs.GCSFileSystem')
    def test_gcsfs_client_property(self, mock_gcsfs_class):
        """Test GCSFS client property."""
        mock_gcsfs = MagicMock()
        mock_gcsfs_class.return_value = mock_gcsfs
        
        loader = GCSLoader(project="test-project")
        client = loader.gcsfs_client
        
        assert client == mock_gcsfs
        mock_gcsfs_class.assert_called_once_with(project="test-project")
    
    def test_check_exists_invalid_path(self):
        """Test check_exists with invalid GCS path."""
        loader = GCSLoader()
        
        result = loader.check_exists("invalid-path")
        assert result is False
    
    @patch('services.common.gcs_loader.gcsfs.GCSFileSystem')
    def test_check_exists_valid_path(self, mock_gcsfs_class):
        """Test check_exists with valid GCS path."""
        mock_gcsfs = MagicMock()
        mock_gcsfs.exists.return_value = True
        mock_gcsfs_class.return_value = mock_gcsfs
        
        loader = GCSLoader()
        result = loader.check_exists("gs://test-bucket/test-path")
        
        assert result is True
        mock_gcsfs.exists.assert_called_once_with("test-bucket/test-path")
    
    def test_clear_cache_disabled(self, temp_dir: Path):
        """Test clear_cache when cache is disabled."""
        loader = GCSLoader(cache_dir=temp_dir, enable_cache=False)
        
        # Should not raise exception, just log warning
        loader.clear_cache()
    
    def test_clear_cache_specific_path(self, temp_dir: Path):
        """Test clearing cache for specific GCS path."""
        loader = GCSLoader(cache_dir=temp_dir, enable_cache=True)
        
        # Create a mock cache directory
        import hashlib
        gcs_path = "gs://test-bucket/test-model"
        path_hash = hashlib.md5(gcs_path.encode()).hexdigest()
        cache_path = temp_dir / path_hash
        cache_path.mkdir(parents=True, exist_ok=True)
        
        assert cache_path.exists()
        
        loader.clear_cache(gcs_path)
        
        assert not cache_path.exists()
    
    def test_clear_entire_cache(self, temp_dir: Path):
        """Test clearing entire cache."""
        loader = GCSLoader(cache_dir=temp_dir, enable_cache=True)
        
        # Create some mock cache files
        (temp_dir / "file1").touch()
        (temp_dir / "file2").touch()
        
        loader.clear_cache()
        
        assert temp_dir.exists()  # Directory should be recreated
        assert len(list(temp_dir.iterdir())) == 0  # But should be empty


@pytest.mark.unit
class TestGCSUtils:
    """Test GCS utility functions."""
    
    def test_validate_gcs_path_valid(self):
        """Test validating valid GCS paths."""
        valid_paths = [
            "gs://bucket/path",
            "gs://my-bucket/model/weights",
            "gs://bucket123/path/to/model.bin"
        ]
        
        for path in valid_paths:
            assert validate_gcs_path(path) is True
    
    def test_validate_gcs_path_invalid(self):
        """Test validating invalid GCS paths."""
        invalid_paths = [
            "",
            "gs://",
            "s3://bucket/path",
            "/local/path",
            "https://example.com"
        ]
        
        for path in invalid_paths:
            assert validate_gcs_path(path) is False
    
    def test_extract_bucket_and_path(self):
        """Test extracting bucket and path from GCS URL."""
        test_cases = [
            ("gs://bucket/path", ("bucket", "path")),
            ("gs://my-bucket/model/weights.bin", ("my-bucket", "model/weights.bin")),
            ("gs://bucket", ("bucket", ""))
        ]
        
        for gcs_path, expected in test_cases:
            bucket, path = extract_bucket_and_path(gcs_path)
            assert (bucket, path) == expected
    
    def test_extract_bucket_and_path_invalid(self):
        """Test extracting from invalid GCS path."""
        with pytest.raises(ValueError, match="Invalid GCS path"):
            extract_bucket_and_path("invalid-path")


@pytest.mark.unit
def test_get_gcs_loader_singleton():
    """Test that get_gcs_loader returns singleton instance."""
    loader1 = get_gcs_loader()
    loader2 = get_gcs_loader()
    
    assert loader1 is loader2