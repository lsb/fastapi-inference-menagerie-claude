"""Update command for existing deployments."""

from typing import Optional

import typer
from rich.console import Console

from cli.utils import (
    validate_model_name, run_command, get_user_prefix, 
    check_k8s_connection, get_git_sha
)

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model service to update"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Resource prefix"),
    registry: str = typer.Option("gcr.io/model-zoo", "--registry", "-r", help="Docker registry"),
    image_tag: Optional[str] = typer.Option(None, "--image-tag", help="Specific image tag to deploy"),
) -> None:
    """Update existing deployment with new image."""
    
    # Validate model name
    if not validate_model_name(model_name):
        console.print(f"[red]Invalid model name: {model_name}[/red]")
        raise typer.Exit(1)
    
    # Check Kubernetes connection
    if not check_k8s_connection():
        console.print("[red]Cannot connect to Kubernetes cluster[/red]")
        raise typer.Exit(1)
    
    # Get defaults
    if prefix is None:
        prefix = get_user_prefix()
    
    if namespace is None:
        namespace = f"{prefix}-zoo"
    
    if image_tag is None:
        sha = get_git_sha()
        image_tag = f"{registry}/{prefix}-{model_name}:gpu-{sha}"
    
    deployment_name = f"{prefix}-{model_name}"
    
    console.print(f"[blue]Updating deployment: {deployment_name}[/blue]")
    console.print(f"[dim]Namespace: {namespace}[/dim]")
    console.print(f"[dim]New image: {image_tag}[/dim]")
    
    try:
        # Update deployment image
        run_command([
            "kubectl", "set", "image",
            "-n", namespace,
            f"deployment/{deployment_name}",
            f"{model_name}={image_tag}"
        ])
        
        console.print(f"[green]✓ Updated deployment image[/green]")
        
        # Wait for rollout to complete
        console.print("[blue]Waiting for rollout to complete...[/blue]")
        run_command([
            "kubectl", "rollout", "status",
            "-n", namespace,
            f"deployment/{deployment_name}",
            "--timeout=300s"
        ])
        
        console.print(f"[green]✓ Successfully updated {model_name}![/green]")
        
    except Exception as e:
        console.print(f"[red]Update failed: {e}[/red]")
        raise typer.Exit(1)