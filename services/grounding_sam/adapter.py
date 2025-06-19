"""Grounding DINO + SAM2 adapter."""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    # SAM2 not available, will use placeholder
    build_sam2 = None
    SAM2ImagePredictor = None

from services.common.adapter import ModelAdapter
from services.common.utils import decode_base64_to_image, encode_image_to_base64

logger = logging.getLogger(__name__)


class GroundingSAMAdapter(ModelAdapter):
    """Grounding DINO + SAM2 adapter for object detection and segmentation."""
    
    def __init__(self, gcs_path: str, device: str) -> None:
        """Initialize Grounding DINO + SAM2 adapter."""
        super().__init__(gcs_path, device)
        self.grounding_model = None
        self.grounding_processor = None
        self.sam_predictor = None
    
    async def load_model(self) -> None:
        """Load Grounding DINO and SAM2 models."""
        logger.info("Loading Grounding DINO + SAM2 models...")
        
        try:
            # Load Grounding DINO
            grounding_model_name = "IDEA-Research/grounding-dino-base"
            self.grounding_processor = AutoProcessor.from_pretrained(grounding_model_name)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                grounding_model_name
            )
            self.grounding_model = self.grounding_model.to(self.device)
            self.grounding_model.eval()
            
            # Load SAM2 (if available)
            if build_sam2 and SAM2ImagePredictor:
                sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"  # TODO: Download from GCS
                model_cfg = "sam2_hiera_l.yaml"
                
                try:
                    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
                    self.sam_predictor = SAM2ImagePredictor(sam2_model)
                    logger.info("SAM2 loaded successfully")
                except Exception as e:
                    logger.warning(f"SAM2 not available: {e}. Using dummy segmentation.")
                    self.sam_predictor = None
            else:
                logger.warning("SAM2 not installed. Using bounding boxes only.")
                self.sam_predictor = None
            
            self._loaded = True
            logger.info("Grounding DINO + SAM2 models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run object detection and segmentation.
        
        Args:
            payload: Input containing:
                - image: base64 encoded image
                - text: text prompt for object detection
                - confidence_threshold: detection confidence (default: 0.3)
                - include_masks: whether to generate masks (default: True)
                
        Returns:
            Dictionary with detections, boxes, and masks
        """
        self._ensure_loaded()
        
        image_b64 = payload['image']
        text_prompt = payload['text']
        confidence_threshold = payload.get('confidence_threshold', 0.3)
        include_masks = payload.get('include_masks', True)
        
        # Decode image
        image = decode_base64_to_image(image_b64)
        
        # Run Grounding DINO
        detections = await self._run_grounding_dino(image, text_prompt, confidence_threshold)
        
        # Run SAM2 if requested and available
        if include_masks and self.sam_predictor and detections['boxes']:
            masks = await self._run_sam2(image, detections['boxes'])
            detections['masks'] = masks
        
        return detections
    
    async def _run_grounding_dino(
        self, 
        image: Image.Image, 
        text: str, 
        threshold: float
    ) -> Dict[str, Any]:
        """Run Grounding DINO object detection."""
        with torch.no_grad():
            inputs = self.grounding_processor(
                images=image,
                text=text,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.grounding_model(**inputs)
            
            # Post-process results
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=[image.size[::-1]]  # (height, width)
            )[0]
            
            # Convert to serializable format
            boxes = results["boxes"].cpu().numpy().tolist()
            scores = results["scores"].cpu().numpy().tolist()
            labels = results["labels"]
            
            return {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "count": len(boxes),
                "text_prompt": text
            }
    
    async def _run_sam2(self, image: Image.Image, boxes: List[List[float]]) -> List[str]:
        """Run SAM2 segmentation on detected boxes."""
        if not self.sam_predictor:
            return []
        
        # Convert PIL to numpy array
        image_array = np.array(image)
        
        # Set image for SAM2
        self.sam_predictor.set_image(image_array)
        
        masks = []
        for box in boxes:
            # Convert box format [x1, y1, x2, y2] to SAM2 format
            input_box = np.array(box)
            
            # Predict mask
            mask, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            # Convert mask to base64
            mask_image = Image.fromarray((mask[0] * 255).astype(np.uint8))
            mask_b64 = encode_image_to_base64(mask_image)
            masks.append(mask_b64)
        
        return masks