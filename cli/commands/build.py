"""Build command for Docker images."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from cli.utils import get_project_root, validate_model_name, run_command, get_user_prefix

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model service to build"),
    registry: str = typer.Option("gcr.io/model-zoo", "--registry", "-r", help="Docker registry"),
    cuda: Optional[str] = typer.Option("12.4", "--cuda", help="CUDA version for GPU builds"),
    cpu: bool = typer.Option(False, "--cpu", help="Build CPU-only image"),
    gpu: bool = typer.Option(True, "--gpu", help="Build GPU image"),
    pytorch_version: Optional[str] = typer.Option(None, "--pytorch", help="PyTorch version"),
    extra_deps: Optional[str] = typer.Option(None, "--extra-deps", help="Extra pip dependencies"),
    push: bool = typer.Option(False, "--push", help="Push image after build"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Resource prefix"),
) -> None:
    """Build Docker image for a model."""
    
    # Validate model name
    if not validate_model_name(model_name):
        console.print(f"[red]Invalid model name: {model_name}[/red]")
        raise typer.Exit(1)
    
    # Get prefix
    if prefix is None:
        prefix = get_user_prefix()
    
    project_root = get_project_root()
    service_dir = project_root / "services" / model_name.lower()
    
    # Check if service exists
    if not service_dir.exists():
        console.print(f"[red]Service not found: {service_dir}[/red]")
        console.print(f"[yellow]Run 'zoo init {model_name}' first[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Building Docker image for {model_name}[/blue]")
    console.print(f"[dim]Service directory: {service_dir}[/dim]")
    console.print(f"[dim]Registry: {registry}[/dim]")
    console.print(f"[dim]Prefix: {prefix}[/dim]")
    
    # Build command arguments
    build_cmd = [
        "python", str(project_root / "docker" / "build.py"),
        model_name,
        "--registry", registry,
        "--prefix", prefix,
    ]
    
    if cpu:
        build_cmd.append("--cpu")
    if gpu:
        build_cmd.append("--gpu")
    if cuda:
        build_cmd.extend(["--base-tag", f"nvcr.io/nvidia/pytorch:24.03-py3"])
    if pytorch_version:
        build_cmd.extend(["--pytorch-version", pytorch_version])
    if extra_deps:
        build_cmd.extend(["--extra-depends", extra_deps])
    if push:
        build_cmd.append("--push")
    
    try:
        # Run build script
        run_command(build_cmd, cwd=project_root)
        
        console.print(f"[green]✓ Successfully built Docker image for {model_name}![/green]")
        
        if push:
            console.print(f"[green]✓ Image pushed to {registry}[/green]")
        else:
            console.print(f"[yellow]To push image, run: zoo build {model_name} --push[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Build failed: {e}[/red]")
        raise typer.Exit(1)