"""Delete command for removing deployments."""

from typing import Optional

import typer
from rich.console import Console

from cli.utils import validate_model_name, run_command, get_user_prefix, check_k8s_connection

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model service to delete"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Resource prefix"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Delete model deployment."""
    
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
    
    deployment_name = f"{prefix}-{model_name}"
    
    console.print(f"[yellow]About to delete deployment: {deployment_name}[/yellow]")
    console.print(f"[dim]Namespace: {namespace}[/dim]")
    
    # Confirmation prompt
    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to delete {model_name}?",
            default=False
        )
        if not confirm:
            console.print("[yellow]Deletion cancelled[/yellow]")
            return
    
    try:
        # Delete resources
        resources_to_delete = [
            f"deployment/{deployment_name}",
            f"service/{deployment_name}",
            f"hpa/{deployment_name}",
        ]
        
        for resource in resources_to_delete:
            try:
                run_command([
                    "kubectl", "delete", 
                    "-n", namespace,
                    resource
                ], check=False)
                console.print(f"[green]✓ Deleted {resource}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not delete {resource}: {e}[/yellow]")
        
        console.print(f"[green]✓ Successfully deleted {model_name} deployment![/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to delete deployment: {e}[/red]")
        raise typer.Exit(1)