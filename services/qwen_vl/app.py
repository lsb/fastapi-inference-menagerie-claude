"""Qwen 2.5 VL FastAPI service."""

import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from services.common.app import create_app
from services.common.config import get_config
from services.common.logging import setup_logging
from services.qwen_vl.adapter import QwenVLAdapter

# Setup logging
setup_logging(service_name="qwen-vl")
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize adapter
adapter = QwenVLAdapter(
    gcs_path=config.model_gcs_path,
    device=config.device
)

# Create FastAPI app
app = create_app(
    model_adapter=adapter,
    title="Qwen 2.5 VL Service",
    description="Visual Question Answering using Qwen 2.5 VL",
    version="1.0.0"
)


# Request/Response models
class VQARequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    question: str = Field(..., description="Question about the image")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature")
    stream: Optional[bool] = Field(False, description="Whether to stream response")


# Routes
@app.post("/v1/vqa/ask")
async def visual_question_answering(request: VQARequest):
    """Visual question answering endpoint."""
    try:
        payload = {
            "image": request.image,
            "question": request.question,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        if request.stream:
            # Return streaming response
            async def generate():
                try:
                    async for token in adapter.stream(payload):
                        yield {"data": token}
                except Exception as e:
                    logger.error(f"Streaming VQA failed: {e}")
                    yield {"data": f"[ERROR] {str(e)}"}
            
            return EventSourceResponse(generate())
        else:
            # Return single response
            result = await adapter.predict(payload)
            return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"VQA failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/vqa/chat")
async def visual_chat(request: VQARequest) -> Dict[str, Any]:
    """Visual chat endpoint (non-streaming)."""
    try:
        payload = {
            "image": request.image,
            "question": request.question,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        result = await adapter.predict(payload)
        return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"Visual chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)