"""Grounding DINO + SAM2 FastAPI service."""

import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.common.app import create_app
from services.common.config import get_config
from services.common.logging import setup_logging
from services.grounding_sam.adapter import GroundingSAMAdapter

# Setup logging
setup_logging(service_name="grounding-sam")
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize adapter
adapter = GroundingSAMAdapter(
    gcs_path=config.model_gcs_path,
    device=config.device
)

# Create FastAPI app
app = create_app(
    model_adapter=adapter,
    title="Grounding DINO + SAM2 Service",
    description="Object detection and segmentation using Grounding DINO and SAM2",
    version="1.0.0"
)


# Request/Response models
class SegmentRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    text: str = Field(..., description="Text prompt for object detection")
    confidence_threshold: Optional[float] = Field(0.3, description="Detection confidence threshold")
    include_masks: Optional[bool] = Field(True, description="Whether to generate segmentation masks")


# Routes
@app.post("/v1/ground/segment")
async def segment_objects(request: SegmentRequest) -> Dict[str, Any]:
    """Detect and segment objects in image."""
    try:
        payload = {
            "image": request.image,
            "text": request.text,
            "confidence_threshold": request.confidence_threshold,
            "include_masks": request.include_masks
        }
        
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ground/detect")
async def detect_objects(request: SegmentRequest) -> Dict[str, Any]:
    """Detect objects in image (bounding boxes only)."""
    try:
        payload = {
            "image": request.image,
            "text": request.text,
            "confidence_threshold": request.confidence_threshold,
            "include_masks": False
        }
        
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)