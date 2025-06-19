"""GCS weight loader for model files."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

import gcsfs
from google.cloud import storage

from services.common.utils import validate_gcs_path, extract_bucket_and_path

logger = logging.getLogger(__name__)


class GCSLoader:
    """Loader for downloading model weights from Google Cloud Storage."""
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cache: bool = True,
        project: Optional[str] = None,
    ) -> None:
        """Initialize GCS loader.
        
        Args:
            cache_dir: Local cache directory for downloaded weights
            enable_cache: Whether to enable local caching
            project: GCP project ID (optional)
        """
        self.enable_cache = enable_cache
        self.project = project
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path("/var/cache/zoo")
        
        self.cache_dir = Path(cache_dir)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self._gcs_client = None
        self._gcsfs_client = None
        
        logger.info(f"Initialized GCS loader with cache: {self.cache_dir}")
    
    @property
    def gcs_client(self) -> storage.Client:
        """Get or create GCS client."""
        if self._gcs_client is None:
            self._gcs_client = storage.Client(project=self.project)
        return self._gcs_client
    
    @property
    def gcsfs_client(self) -> gcsfs.GCSFileSystem:
        """Get or create GCSFS client."""
        if self._gcsfs_client is None:
            self._gcsfs_client = gcsfs.GCSFileSystem(project=self.project)
        return self._gcsfs_client
    
    def download_weights(
        self,
        gcs_path: str,
        local_path: Optional[Union[str, Path]] = None,
        force_download: bool = False,
    ) -> Path:
        """Download model weights from GCS.
        
        Args:
            gcs_path: GCS path to model weights (gs://bucket/path)
            local_path: Local path to download to (optional)
            force_download: Force download even if cached
            
        Returns:
            Path to downloaded weights directory
        """
        if not validate_gcs_path(gcs_path):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        # Determine local path
        if local_path is None:
            # Use cache directory with hashed path
            import hashlib
            path_hash = hashlib.md5(gcs_path.encode()).hexdigest()
            local_path = self.cache_dir / path_hash
        
        local_path = Path(local_path)
        
        # Check if already cached
        if not force_download and local_path.exists() and self.enable_cache:
            logger.info(f"Using cached weights: {local_path}")
            return local_path
        
        logger.info(f"Downloading weights from {gcs_path} to {local_path}")
        
        # Create local directory
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download using gcsfs for better streaming support
            bucket, object_path = extract_bucket_and_path(gcs_path)
            
            if self.gcsfs_client.isdir(f"{bucket}/{object_path}"):
                # Download directory
                self._download_directory(f"{bucket}/{object_path}", local_path)
            else:
                # Download single file
                self._download_file(f"{bucket}/{object_path}", local_path)
            
            logger.info(f"Successfully downloaded weights to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download weights from {gcs_path}: {e}")
            # Clean up partial download
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
            raise
    
    def _download_directory(self, gcs_dir: str, local_dir: Path) -> None:
        """Download entire directory from GCS."""
        logger.info(f"Downloading directory: gs://{gcs_dir}")
        
        # List all files in the directory
        files = self.gcsfs_client.find(gcs_dir)
        
        for file_path in files:
            if self.gcsfs_client.isfile(file_path):
                # Get relative path
                rel_path = file_path[len(gcs_dir):].lstrip('/')
                local_file = local_dir / rel_path
                
                # Create parent directories
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                self.gcsfs_client.get(file_path, str(local_file))
                logger.debug(f"Downloaded: {rel_path}")
    
    def _download_file(self, gcs_file: str, local_dir: Path) -> None:
        """Download single file from GCS."""
        logger.info(f"Downloading file: gs://{gcs_file}")
        
        # Get filename
        filename = gcs_file.split('/')[-1]
        local_file = local_dir / filename
        
        # Download file
        self.gcsfs_client.get(gcs_file, str(local_file))
        logger.debug(f"Downloaded: {filename}")
    
    def upload_weights(
        self,
        local_path: Union[str, Path],
        gcs_path: str,
        recursive: bool = True,
    ) -> None:
        """Upload model weights to GCS.
        
        Args:
            local_path: Local path to weights
            gcs_path: Target GCS path
            recursive: Upload directory recursively
        """
        if not validate_gcs_path(gcs_path):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")
        
        logger.info(f"Uploading weights from {local_path} to {gcs_path}")
        
        bucket, object_path = extract_bucket_and_path(gcs_path)
        
        try:
            if local_path.is_dir() and recursive:
                # Upload directory
                self._upload_directory(local_path, f"{bucket}/{object_path}")
            else:
                # Upload single file
                self._upload_file(local_path, f"{bucket}/{object_path}")
            
            logger.info(f"Successfully uploaded weights to {gcs_path}")
            
        except Exception as e:
            logger.error(f"Failed to upload weights to {gcs_path}: {e}")
            raise
    
    def _upload_directory(self, local_dir: Path, gcs_dir: str) -> None:
        """Upload entire directory to GCS."""
        logger.info(f"Uploading directory: {local_dir}")
        
        for local_file in local_dir.rglob('*'):
            if local_file.is_file():
                # Get relative path
                rel_path = local_file.relative_to(local_dir)
                gcs_file = f"{gcs_dir}/{rel_path}"
                
                # Upload file
                self.gcsfs_client.put(str(local_file), gcs_file)
                logger.debug(f"Uploaded: {rel_path}")
    
    def _upload_file(self, local_file: Path, gcs_file: str) -> None:
        """Upload single file to GCS."""
        logger.info(f"Uploading file: {local_file}")
        self.gcsfs_client.put(str(local_file), gcs_file)
    
    def check_exists(self, gcs_path: str) -> bool:
        """Check if GCS path exists.
        
        Args:
            gcs_path: GCS path to check
            
        Returns:
            True if path exists
        """
        if not validate_gcs_path(gcs_path):
            return False
        
        try:
            bucket, object_path = extract_bucket_and_path(gcs_path)
            return self.gcsfs_client.exists(f"{bucket}/{object_path}")
        except Exception:
            return False
    
    def clear_cache(self, gcs_path: Optional[str] = None) -> None:
        """Clear local cache.
        
        Args:
            gcs_path: Specific GCS path to clear (optional)
        """
        if not self.enable_cache:
            logger.warning("Cache is disabled")
            return
        
        if gcs_path:
            # Clear specific cache entry
            import hashlib
            path_hash = hashlib.md5(gcs_path.encode()).hexdigest()
            cache_path = self.cache_dir / path_hash
            
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for {gcs_path}")
        else:
            # Clear entire cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire cache")


# Singleton instance
_gcs_loader = None


def get_gcs_loader(
    cache_dir: Optional[Union[str, Path]] = None,
    enable_cache: bool = True,
    project: Optional[str] = None,
) -> GCSLoader:
    """Get singleton GCS loader instance."""
    global _gcs_loader
    
    if _gcs_loader is None:
        _gcs_loader = GCSLoader(
            cache_dir=cache_dir,
            enable_cache=enable_cache,
            project=project,
        )
    
    return _gcs_loader