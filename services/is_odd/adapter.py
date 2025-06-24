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
        
        # Validate input
        if not isinstance(number, (int, float)):
            raise ValueError(f"Number must be int or float, got {type(number)}")
        
        # Convert to int for odd/even check
        num_int = int(number)
        
        # The actual "inference"
        is_odd = bool(num_int % 2)
        
        return {
            "is_odd": is_odd,
            "number": num_int,
            "computation": f"{num_int} % 2 = {num_int % 2}"
        }