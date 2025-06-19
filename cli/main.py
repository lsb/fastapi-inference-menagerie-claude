"""Main CLI entry point for the model zoo."""

import os
from typing import Optional

import typer
from rich.console import Console

from cli.commands import init, build, deploy, logs, delete, update, rollback

app = typer.Typer(
    name="zoo",
    help="ML Model Zoo CLI for FastAPI + Kubernetes deployments",
    rich_markup_mode="rich",
)

console = Console()

# Add subcommands
app.add_typer(init.app, name="init", help="Initialize a new model service")
app.add_typer(build.app, name="build", help="Build Docker image for a model")
app.add_typer(deploy.app, name="deploy", help="Deploy model to Kubernetes")
app.add_typer(update.app, name="update", help="Update existing deployment")
app.add_typer(rollback.app, name="rollback", help="Rollback to previous version")
app.add_typer(logs.app, name="logs", help="View model service logs")
app.add_typer(delete.app, name="delete", help="Delete model deployment")


@app.callback()
def main(
    prefix: Optional[str] = typer.Option(
        None,
        "--prefix",
        help="Resource prefix (defaults to $USER)",
        envvar="ZOO_PREFIX",
    ),
    namespace: Optional[str] = typer.Option(
        None,
        "--namespace", 
        help="Kubernetes namespace (defaults to $PREFIX-zoo)",
        envvar="ZOO_NAMESPACE",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """ML Model Zoo CLI for FastAPI + Kubernetes deployments."""
    # Set global context
    if prefix is None:
        prefix = os.getenv("USER", "default")
    
    if namespace is None:
        namespace = f"{prefix}-zoo"
    
    # Store in app state for subcommands
    app.state = {
        "prefix": prefix,
        "namespace": namespace,
        "verbose": verbose,
    }


if __name__ == "__main__":
    app()