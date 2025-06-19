"""Base ModelAdapter class for all ML models."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Union, Optional
from pathlib import Path

from services.common.gcs_loader import get_gcs_loader

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    def __init__(self, gcs_path: str, device: str) -> None:
        """Initialize the model adapter.
        
        Args:
            gcs_path: GCS path to model weights (gs://bucket/path)
            device: Device to run model on (cuda:0, cpu, etc.)
        """
        self.gcs_path = gcs_path
        self.device = device
        self.model = None
        self._loaded = False
        self._local_weights_path: Optional[Path] = None
        
        # Initialize GCS loader
        self.gcs_loader = get_gcs_loader()
        
        logger.info(f"Initializing adapter for {self.__class__.__name__}")
        logger.info(f"GCS Path: {gcs_path}")
        logger.info(f"Device: {device}")
    
    async def download_weights(self, force_download: bool = False) -> Path:
        """Download model weights from GCS.
        
        Args:
            force_download: Force download even if cached
            
        Returns:
            Path to local weights directory
        """
        if self._local_weights_path and not force_download:
            return self._local_weights_path
        
        logger.info(f"Downloading weights from {self.gcs_path}")
        
        # Check if GCS path exists
        if not self.gcs_loader.check_exists(self.gcs_path):
            logger.warning(f"GCS path does not exist: {self.gcs_path}")
            logger.info("Proceeding without downloading weights (may use HuggingFace Hub)")
            return None
        
        # Download weights asynchronously
        loop = asyncio.get_event_loop()
        self._local_weights_path = await loop.run_in_executor(
            None,
            self.gcs_loader.download_weights,
            self.gcs_path,
            None,  # Use default cache path
            force_download
        )
        
        logger.info(f"Weights downloaded to: {self._local_weights_path}")
        return self._local_weights_path
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load model from GCS path."""
        pass
    
    @abstractmethod
    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the model.
        
        Args:
            payload: Input data for inference
            
        Returns:
            Prediction results
        """
        pass
    
    async def stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream prediction results (for generative models).
        
        Args:
            payload: Input data for inference
            
        Yields:
            Streaming prediction tokens/results
        """
        # Default implementation for non-streaming models
        result = await self.predict(payload)
        yield str(result)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy" if self._loaded else "loading",
            "model_class": self.__class__.__name__,
            "device": self.device,
            "gcs_path": self.gcs_path,
        }
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded