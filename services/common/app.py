"""Common FastAPI application core."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Callable

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sse_starlette.sse import EventSourceResponse

from services.common.adapter import ModelAdapter
from services.common.metrics import MetricsExporter

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration')
QUEUE_DEPTH = Gauge('model_queue_depth', 'Current queue depth')
MODEL_LOAD_TIME = Histogram('model_load_time_seconds', 'Model loading time')

# Global queue depth counter
_queue_depth = 0
_queue_lock = asyncio.Lock()


async def increment_queue_depth() -> None:
    """Increment queue depth counter."""
    global _queue_depth
    async with _queue_lock:
        _queue_depth += 1
        QUEUE_DEPTH.set(_queue_depth)


async def decrement_queue_depth() -> None:
    """Decrement queue depth counter."""
    global _queue_depth
    async with _queue_lock:
        _queue_depth = max(0, _queue_depth - 1)
        QUEUE_DEPTH.set(_queue_depth)


def create_app(
    model_adapter: ModelAdapter,
    title: str = "Model Zoo Service",
    description: str = "FastAPI service for ML model inference",
    version: str = "1.0.0",
) -> FastAPI:
    """Create FastAPI application with common middleware and routes.
    
    Args:
        model_adapter: Model adapter instance
        title: API title
        description: API description
        version: API version
        
    Returns:
        Configured FastAPI application
    """
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup and shutdown events."""
        # Startup
        logger.info("Starting up model service...")
        start_time = time.time()
        
        try:
            await model_adapter.load_model()
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.observe(load_time)
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Start metrics exporter
        metrics_exporter = MetricsExporter()
        metrics_task = asyncio.create_task(metrics_exporter.start())
        
        yield
        
        # Shutdown
        logger.info("Shutting down model service...")
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def add_queue_depth_metric(request: Request, call_next: Callable) -> Response:
        """Middleware to track queue depth and request metrics."""
        await increment_queue_depth()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            return response
        
        except Exception as e:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            raise
        
        finally:
            await decrement_queue_depth()
    
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return await model_adapter.health_check()
    
    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    
    @app.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint."""
        return {"message": f"Welcome to {title}"}
    
    # Store adapter in app state for use in model-specific routes
    app.state.model_adapter = model_adapter
    
    return app