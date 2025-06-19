"""Prometheus metrics exporter."""

import asyncio
import logging
import sys
from prometheus_client import generate_latest

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Exports Prometheus metrics to stdout every 5 seconds."""
    
    def __init__(self, interval: int = 5) -> None:
        """Initialize metrics exporter.
        
        Args:
            interval: Export interval in seconds
        """
        self.interval = interval
        self._running = False
    
    async def start(self) -> None:
        """Start metrics export loop."""
        self._running = True
        logger.info(f"Starting metrics exporter (interval: {self.interval}s)")
        
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                if self._running:  # Check again after sleep
                    await self._export_metrics()
            except asyncio.CancelledError:
                logger.info("Metrics exporter cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics export: {e}")
    
    async def _export_metrics(self) -> None:
        """Export current metrics to stdout."""
        try:
            metrics_data = generate_latest().decode('utf-8')
            
            # Write metrics to stdout with timestamp
            print(f"# METRICS EXPORT {asyncio.get_event_loop().time()}")
            print(metrics_data)
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def stop(self) -> None:
        """Stop metrics export loop."""
        self._running = False