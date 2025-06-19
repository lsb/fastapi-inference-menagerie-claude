"""Logging configuration for model services."""

import logging
import sys
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Structured log formatter for model services."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured output."""
        # Base format: [timestamp] level=model INFO message="..."
        timestamp = self.formatTime(record)
        level = record.levelname
        
        # Extract extra fields from record
        extra_fields = []
        if hasattr(record, 'duration_ms'):
            extra_fields.append(f"duration_ms={record.duration_ms}")
        if hasattr(record, 'queue_depth'):
            extra_fields.append(f"queue_depth={record.queue_depth}")
        if hasattr(record, 'model_name'):
            extra_fields.append(f"model_name={record.model_name}")
        
        # Build message
        message = record.getMessage()
        if extra_fields:
            message = f"{message} {' '.join(extra_fields)}"
        
        return f"[{timestamp}] level=model {level} message=\"{message}\""


def setup_logging(
    level: str = "INFO",
    service_name: Optional[str] = None,
) -> None:
    """Setup structured logging for model services.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        service_name: Name of the service for log context
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(message)s",
    )
    
    # Apply structured formatter
    formatter = StructuredFormatter()
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    
    # Set service name if provided
    if service_name:
        logging.getLogger().info(f"Starting service: {service_name}")
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)