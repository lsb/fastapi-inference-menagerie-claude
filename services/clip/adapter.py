"""CLIP model adapter."""

import logging
from typing import Dict, Any, List, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from services.common.adapter import ModelAdapter
from services.common.utils import decode_base64_to_image

logger = logging.getLogger(__name__)


class CLIPAdapter(ModelAdapter):
    """CLIP model adapter for text and image encoding."""
    
    def __init__(self, gcs_path: str, device: str) -> None:
        """Initialize CLIP adapter."""
        super().__init__(gcs_path, device)
        self.model: CLIPModel = None
        self.processor: CLIPProcessor = None
    
    async def load_model(self) -> None:
        """Load CLIP model from GCS path."""
        logger.info("Loading CLIP model...")
        
        try:
            # Try to download weights from GCS first
            weights_path = await self.download_weights()
            
            if weights_path and weights_path.exists():
                # Load from local weights
                logger.info(f"Loading CLIP from local weights: {weights_path}")
                self.processor = CLIPProcessor.from_pretrained(str(weights_path))
                self.model = CLIPModel.from_pretrained(str(weights_path))
            else:
                # Fallback to HuggingFace Hub
                logger.info("Loading CLIP from HuggingFace Hub")
                model_name = "openai/clip-vit-base-patch32"
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model = CLIPModel.from_pretrained(model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            logger.info(f"CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run CLIP inference.
        
        Args:
            payload: Input containing 'texts' and/or 'images' keys
                    - texts: List of strings
                    - images: List of base64 encoded images or PIL Images
                    - task: 'encode_text', 'encode_image', or 'similarity'
                    
        Returns:
            Dictionary with embeddings or similarity scores
        """
        self._ensure_loaded()
        
        task = payload.get('task', 'encode_text')
        
        if task == 'encode_text':
            return await self._encode_text(payload['texts'])
        elif task == 'encode_image':
            return await self._encode_images(payload['images'])
        elif task == 'similarity':
            return await self._compute_similarity(payload)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    async def _encode_text(self, texts: List[str]) -> Dict[str, Any]:
        """Encode text inputs."""
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            return {
                "embeddings": text_features.cpu().numpy().tolist(),
                "shape": list(text_features.shape),
                "count": len(texts)
            }
    
    async def _encode_images(self, images: List[Union[str, Image.Image]]) -> Dict[str, Any]:
        """Encode image inputs."""
        # Convert base64 strings to PIL Images if needed
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(decode_base64_to_image(img))
            else:
                pil_images.append(img)
        
        with torch.no_grad():
            inputs = self.processor(
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            return {
                "embeddings": image_features.cpu().numpy().tolist(),
                "shape": list(image_features.shape),
                "count": len(images)
            }
    
    async def _compute_similarity(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compute text-image similarity."""
        texts = payload.get('texts', [])
        images = payload.get('images', [])
        
        if not texts or not images:
            raise ValueError("Both 'texts' and 'images' required for similarity task")
        
        # Get embeddings
        text_result = await self._encode_text(texts)
        image_result = await self._encode_images(images)
        
        # Compute similarity matrix
        text_embeddings = torch.tensor(text_result['embeddings'])
        image_embeddings = torch.tensor(image_result['embeddings'])
        
        similarity_matrix = torch.matmul(text_embeddings, image_embeddings.T)
        
        return {
            "similarity_matrix": similarity_matrix.numpy().tolist(),
            "text_count": len(texts),
            "image_count": len(images),
            "texts": texts,
        }