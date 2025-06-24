"""Is-odd model adapter for measuring service overhead."""

import logging
from typing import Dict, Any

from services.common.adapter import ModelAdapter

logger = logging.getLogger(__name__)


class IsOddAdapter(ModelAdapter):
    """Simple is-odd adapter for measuring FastAPI overhead."""
    
    def __init__(self, gcs_path: str = None, device: str = "cpu") -> None:
        """Initialize is-odd adapter."""
        super().__init__(gcs_path, device)
        self.model = "is_odd_function"
    
    async def load_model(self) -> None:
        """Load is-odd 'model' (no-op since it's just a function)."""
        logger.info("Loading is-odd model...")
        
        # Simulate minimal loading time
        import asyncio
        await asyncio.sleep(0.001)
        
        self._loaded = True
        logger.info("Is-odd model loaded successfully")
    
    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if a number is odd.
        
        Args:
            payload: Input data containing 'number' field
            
        Returns:
            Dictionary with 'is_odd' boolean result
        """
        self._ensure_loaded()
        
        if "number" not in payload:
            raise ValueError("Missing 'number' field in payload")
        
        number = payload["number"]
        
        # Handle string inputs (from HTTP) and numeric inputs (from direct calls)
        if isinstance(number, str):
            try:
                num_int = int(number)
            except ValueError:
                raise ValueError(f"Cannot convert string '{number}' to integer")
        elif isinstance(number, (int, float)):
            num_int = int(number)
        else:
            raise ValueError(f"Number must be int, float, or string, got {type(number)}")
        
        # The num_int is ready for computation
        
        # The actual "inference"
        is_odd = bool(num_int % 2)
        
        return {
            "is_odd": is_odd,
            "number": num_int,
            "computation": f"{num_int} % 2 = {num_int % 2}"
        }