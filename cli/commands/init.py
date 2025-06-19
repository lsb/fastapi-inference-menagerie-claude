"""Initialize command for creating new model services."""

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from cli.utils import get_project_root, validate_model_name

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model service to create"),
    template: Optional[str] = typer.Option(
        "basic", "--template", "-t", help="Template to use (basic, clip, grounding-sam, qwen-vl)"
    )
) -> None:
    """Initialize a new model service template."""
    
    # Validate model name
    if not validate_model_name(model_name):
        console.print(f"[red]Invalid model name: {model_name}[/red]")
        console.print("[yellow]Model name must be alphanumeric with hyphens/underscores only[/yellow]")
        raise typer.Exit(1)
    
    project_root = get_project_root()
    service_dir = project_root / "services" / model_name.lower()
    
    # Check if service already exists
    if service_dir.exists():
        console.print(f"[red]Service directory already exists: {service_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Creating service: {model_name}[/blue]")
    console.print(f"[dim]Directory: {service_dir}[/dim]")
    console.print(f"[dim]Template: {template}[/dim]")
    
    # Create service directory
    service_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy template files
    if template == "basic":
        _create_basic_template(service_dir, model_name)
    elif template == "clip":
        _copy_existing_service(project_root / "services" / "clip", service_dir, model_name)
    elif template == "grounding-sam":
        _copy_existing_service(project_root / "services" / "grounding_sam", service_dir, model_name)
    elif template == "qwen-vl":
        _copy_existing_service(project_root / "services" / "qwen_vl", service_dir, model_name)
    else:
        console.print(f"[red]Unknown template: {template}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓ Service {model_name} initialized successfully![/green]")
    console.print(f"[yellow]Next steps:[/yellow]")
    console.print(f"1. Edit {service_dir}/adapter.py to implement your model")
    console.print(f"2. Update {service_dir}/app.py with your API endpoints")
    console.print(f"3. Run: zoo build {model_name}")
    console.print(f"4. Run: zoo deploy {model_name}")


def _create_basic_template(service_dir: Path, model_name: str) -> None:
    """Create basic template files."""
    
    # Create __init__.py
    (service_dir / "__init__.py").write_text("")
    
    # Create adapter.py
    adapter_content = f'''"""{{model_name}} model adapter."""

import logging
from typing import Dict, Any

from services.common.adapter import ModelAdapter

logger = logging.getLogger(__name__)


class {model_name.title().replace("-", "").replace("_", "")}Adapter(ModelAdapter):
    """{{model_name}} model adapter."""
    
    def __init__(self, gcs_path: str, device: str) -> None:
        """Initialize {{model_name}} adapter."""
        super().__init__(gcs_path, device)
        self.model = None
    
    async def load_model(self) -> None:
        """Load {{model_name}} model from GCS path."""
        logger.info("Loading {{model_name}} model...")
        
        try:
            # TODO: Implement model loading
            # self.model = load_your_model(self.gcs_path, device=self.device)
            
            self._loaded = True
            logger.info(f"{{model_name}} model loaded successfully on {{self.device}}")
            
        except Exception as e:
            logger.error(f"Failed to load {{model_name}} model: {{e}}")
            raise
    
    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the {{model_name}} model.
        
        Args:
            payload: Input data for inference
            
        Returns:
            Prediction results
        """
        self._ensure_loaded()
        
        # TODO: Implement inference logic
        # result = self.model(payload)
        
        return {{"message": "TODO: Implement inference for {{model_name}}"}}
'''
    
    (service_dir / "adapter.py").write_text(adapter_content)
    
    # Create app.py
    app_content = f'''"""{model_name} FastAPI service."""

import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.common.app import create_app
from services.common.config import get_config
from services.common.logging import setup_logging
from services.{model_name.lower()}.adapter import {model_name.title().replace("-", "").replace("_", "")}Adapter

# Setup logging
setup_logging(service_name="{model_name}")
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize adapter
adapter = {model_name.title().replace("-", "").replace("_", "")}Adapter(
    gcs_path=config.model_gcs_path,
    device=config.device
)

# Create FastAPI app
app = create_app(
    model_adapter=adapter,
    title="{model_name.title()} Model Service",
    description="{model_name} model inference service",
    version="1.0.0"
)


# Request/Response models
class PredictRequest(BaseModel):
    # TODO: Define your request schema
    input_data: Dict[str, Any] = Field(..., description="Input data for inference")


# Routes
@app.post("/v1/{model_name.lower()}/predict")
async def predict(request: PredictRequest) -> Dict[str, Any]:
    """Run inference on {model_name} model."""
    try:
        result = await adapter.predict(request.input_data)
        return {{"success": True, "result": result}}
    except Exception as e:
        logger.error(f"Prediction failed: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
'''
    
    (service_dir / "app.py").write_text(app_content)


def _copy_existing_service(source_dir: Path, target_dir: Path, model_name: str) -> None:
    """Copy existing service as template."""
    if not source_dir.exists():
        console.print(f"[red]Template source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)
    
    # Copy all files
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    
    console.print(f"[green]✓ Copied template from {source_dir.name}[/green]")
    console.print(f"[yellow]Note: You may need to customize the adapter for your specific model[/yellow]")