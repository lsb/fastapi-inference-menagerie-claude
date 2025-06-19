"""Configuration management for model services."""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class ModelServiceConfig(BaseSettings):
    """Base configuration for model services."""
    
    # Model configuration
    model_gcs_path: str = Field(..., env="MODEL_GCS_PATH")
    model_name: str = Field(..., env="MODEL_NAME")
    device: str = Field(default="cuda:0", env="DEVICE")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Cache configuration
    cache_dir: str = Field(default="/var/cache/zoo", env="CACHE_DIR")
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    
    # GCS configuration
    gcp_project: Optional[str] = Field(default=None, env="GCP_PROJECT")
    gcs_timeout: int = Field(default=300, env="GCS_TIMEOUT")
    
    # Resource limits
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=120, env="REQUEST_TIMEOUT")
    
    # Metrics configuration
    metrics_interval: int = Field(default=5, env="METRICS_INTERVAL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_config() -> ModelServiceConfig:
    """Get configuration instance."""
    return ModelServiceConfig()