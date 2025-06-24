"""Is-odd FastAPI service for measuring overhead."""

import logging
from typing import Dict, Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from services.common.app import create_app
from services.common.config import get_config
from services.common.logging import setup_logging
from services.is_odd.adapter import IsOddAdapter

# Setup logging
setup_logging(service_name="is-odd")
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize adapter
adapter = IsOddAdapter(
    gcs_path=config.model_gcs_path,
    device=config.device
)

# Create FastAPI app
app = create_app(
    model_adapter=adapter,
    title="Is-Odd Demo Service",
    description="Simple demo service to measure FastAPI overhead",
    version="1.0.0"
)


# Request/Response models
class IsOddRequest(BaseModel):
    """Request for is-odd prediction."""
    number: float = Field(..., description="Number to check if odd")


class IsOddResponse(BaseModel):
    """Response from is-odd prediction."""
    success: bool = Field(..., description="Whether the request succeeded")
    result: Dict[str, Any] = Field(..., description="Prediction result")


# Routes
@app.post("/v1/is-odd/predict", response_model=IsOddResponse)
async def predict_is_odd(request: IsOddRequest) -> IsOddResponse:
    """Check if a number is odd."""
    try:
        result = await adapter.predict({"number": request.number})
        return IsOddResponse(success=True, result=result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/is-odd/batch", response_model=IsOddResponse)
async def predict_batch_is_odd(numbers: list[float]) -> IsOddResponse:
    """Check if multiple numbers are odd."""
    try:
        results = []
        for number in numbers:
            result = await adapter.predict({"number": number})
            results.append(result)
        
        return IsOddResponse(
            success=True, 
            result={
                "batch_results": results,
                "count": len(results)
            }
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)