"""Logs command for viewing service logs."""

from typing import Optional

import typer
from rich.console import Console

from cli.utils import validate_model_name, run_command, get_user_prefix, check_k8s_connection

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model service"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Resource prefix"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    tail: int = typer.Option(100, "--tail", help="Number of lines to show from end"),
) -> None:
    """View model service logs."""
    
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
    
    console.print(f"[blue]Viewing logs for {model_name}[/blue]")
    console.print(f"[dim]Namespace: {namespace}[/dim]")
    console.print(f"[dim]Deployment: {deployment_name}[/dim]")
    
    # Build kubectl command
    cmd = [
        "kubectl", "logs",
        "-n", namespace,
        f"deployment/{deployment_name}",
        "--tail", str(tail),
    ]
    
    if follow:
        cmd.append("--follow")
    
    try:
        # Stream logs directly to terminal
        run_command(cmd, capture_output=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Log viewing interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to view logs: {e}[/red]")
        raise typer.Exit(1)