"""Common utilities for model services."""

import base64
import io
import json
from typing import Any, Dict, Union
from PIL import Image
import torch


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image.
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        PIL Image
    """
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))


def tensor_to_json_serializable(obj: Any) -> Any:
    """Convert tensors and other objects to JSON serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: tensor_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [tensor_to_json_serializable(item) for item in obj]
    else:
        return obj


def get_device(prefer_gpu: bool = True) -> str:
    """Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        Device string (cuda:0, cpu, etc.)
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def validate_gcs_path(gcs_path: str) -> bool:
    """Validate GCS path format.
    
    Args:
        gcs_path: GCS path to validate
        
    Returns:
        True if valid GCS path
    """
    return gcs_path.startswith("gs://") and len(gcs_path) > 5


def extract_bucket_and_path(gcs_path: str) -> tuple[str, str]:
    """Extract bucket and path from GCS path.
    
    Args:
        gcs_path: Full GCS path (gs://bucket/path)
        
    Returns:
        Tuple of (bucket_name, object_path)
    """
    if not validate_gcs_path(gcs_path):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    path_parts = gcs_path[5:].split("/", 1)  # Remove gs:// prefix
    bucket = path_parts[0]
    object_path = path_parts[1] if len(path_parts) > 1 else ""
    
    return bucket, object_path