"""CLIP FastAPI service."""

import logging
from typing import Dict, Any, List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.common.app import create_app
from services.common.config import get_config
from services.common.logging import setup_logging
from services.clip.adapter import CLIPAdapter

# Setup logging
setup_logging(service_name="clip")
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize adapter
adapter = CLIPAdapter(
    gcs_path=config.model_gcs_path,
    device=config.device
)

# Create FastAPI app
app = create_app(
    model_adapter=adapter,
    title="CLIP Model Service",
    description="CLIP model for text and image encoding",
    version="1.0.0"
)


# Request/Response models
class TextEncodeRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to encode")


class ImageEncodeRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded images")


class SimilarityRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts")
    images: List[str] = Field(..., description="List of base64 encoded images")


# Routes
@app.post("/v1/clip/encode")
async def encode_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Universal encoding endpoint for CLIP."""
    try:
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/clip/encode/text")
async def encode_text(request: TextEncodeRequest) -> Dict[str, Any]:
    """Encode text inputs."""
    try:
        payload = {"task": "encode_text", "texts": request.texts}
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Text encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/clip/encode/image")
async def encode_images(request: ImageEncodeRequest) -> Dict[str, Any]:
    """Encode image inputs."""
    try:
        payload = {"task": "encode_image", "images": request.images}
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/clip/similarity")
async def compute_similarity(request: SimilarityRequest) -> Dict[str, Any]:
    """Compute text-image similarity."""
    try:
        payload = {
            "task": "similarity",
            "texts": request.texts,
            "images": request.images
        }
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)