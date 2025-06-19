"""Rollback command for reverting deployments."""

from typing import Optional

import typer
from rich.console import Console

from cli.utils import (
    validate_model_name, run_command, get_user_prefix, check_k8s_connection
)

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model service to rollback"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Resource prefix"),
    revision: Optional[int] = typer.Option(None, "--revision", help="Specific revision to rollback to"),
) -> None:
    """Rollback deployment to previous version."""
    
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
    
    console.print(f"[blue]Rolling back deployment: {deployment_name}[/blue]")
    console.print(f"[dim]Namespace: {namespace}[/dim]")
    
    if revision:
        console.print(f"[dim]Target revision: {revision}[/dim]")
    else:
        console.print("[dim]Rolling back to previous revision[/dim]")
    
    try:
        # Show rollout history first
        console.print("[blue]Current rollout history:[/blue]")
        run_command([
            "kubectl", "rollout", "history",
            "-n", namespace,
            f"deployment/{deployment_name}"
        ], capture_output=False)
        
        # Perform rollback
        rollback_cmd = [
            "kubectl", "rollout", "undo",
            "-n", namespace,
            f"deployment/{deployment_name}"
        ]
        
        if revision:
            rollback_cmd.extend([f"--to-revision={revision}"])
        
        run_command(rollback_cmd)
        
        console.print(f"[green]✓ Initiated rollback[/green]")
        
        # Wait for rollout to complete
        console.print("[blue]Waiting for rollback to complete...[/blue]")
        run_command([
            "kubectl", "rollout", "status",
            "-n", namespace,
            f"deployment/{deployment_name}",
            "--timeout=300s"
        ])
        
        console.print(f"[green]✓ Successfully rolled back {model_name}![/green]")
        
    except Exception as e:
        console.print(f"[red]Rollback failed: {e}[/red]")
        raise typer.Exit(1)