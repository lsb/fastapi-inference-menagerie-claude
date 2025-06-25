"""CLIP FastAPI service."""

import logging
from typing import Dict, Any, List, Union, Optional
from PIL import Image
import io

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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


# Routes
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
async def encode_images(images: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Encode image inputs using multipart form data."""
    try:
        # Convert uploaded files to PIL Images
        pil_images = []
        for image_file in images:
            # Read the uploaded file
            contents = await image_file.read()
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(contents))
            pil_images.append(pil_image)
        
        payload = {"task": "encode_image", "images": pil_images}
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/clip/similarity")
async def compute_similarity(
    texts: str = Form(..., description="JSON array of texts to compare"),
    images: List[UploadFile] = File(..., description="Images to compare against texts")
) -> Dict[str, Any]:
    """Compute text-image similarity using multipart form data."""
    try:
        import json
        
        # Parse texts from JSON string
        texts_list = json.loads(texts)
        if not isinstance(texts_list, list):
            raise ValueError("texts must be a JSON array")
        
        # Convert uploaded files to PIL Images
        pil_images = []
        for image_file in images:
            contents = await image_file.read()
            pil_image = Image.open(io.BytesIO(contents))
            pil_images.append(pil_image)
        
        payload = {
            "task": "similarity",
            "texts": texts_list,
            "images": pil_images
        }
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/clip/encode")
async def encode_endpoint(
    task: str = Form(..., description="Task type: encode_text, encode_image, or similarity"),
    texts: Optional[str] = Form(None, description="JSON array of texts (for text/similarity tasks)"),
    images: Optional[List[UploadFile]] = File(None, description="Images (for image/similarity tasks)")
) -> Dict[str, Any]:
    """Universal encoding endpoint for CLIP with multipart support."""
    try:
        import json
        
        if task == "encode_text":
            if not texts:
                raise ValueError("texts parameter required for encode_text task")
            texts_list = json.loads(texts)
            payload = {"task": "encode_text", "texts": texts_list}
            
        elif task == "encode_image":
            if not images:
                raise ValueError("images parameter required for encode_image task")
            pil_images = []
            for image_file in images:
                contents = await image_file.read()
                pil_image = Image.open(io.BytesIO(contents))
                pil_images.append(pil_image)
            payload = {"task": "encode_image", "images": pil_images}
            
        elif task == "similarity":
            if not texts or not images:
                raise ValueError("Both texts and images required for similarity task")
            texts_list = json.loads(texts)
            pil_images = []
            for image_file in images:
                contents = await image_file.read()
                pil_image = Image.open(io.BytesIO(contents))
                pil_images.append(pil_image)
            payload = {"task": "similarity", "texts": texts_list, "images": pil_images}
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)