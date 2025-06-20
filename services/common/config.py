"""Configuration management for model services."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelServiceConfig(BaseSettings):
    """Base configuration for model services."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Model configuration  
    model_gcs_path: Optional[str] = None
    model_name: str = "test-model"
    device: str = "cuda:0"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Logging configuration
    log_level: str = "INFO"
    
    # Cache configuration
    cache_dir: str = "/var/cache/zoo"
    enable_cache: bool = True
    
    # GCS configuration
    gcp_project: Optional[str] = None
    gcs_timeout: int = 300
    
    # Resource limits
    max_concurrent_requests: int = 10
    request_timeout: int = 120
    
    # Metrics configuration
    metrics_interval: int = 5


def get_config() -> ModelServiceConfig:
    """Get configuration instance."""
    return ModelServiceConfig()