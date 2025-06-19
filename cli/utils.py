"""CLI utilities and helper functions."""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json

from rich.console import Console
from rich.table import Table

console = Console()


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_user_prefix() -> str:
    """Get user prefix for resources."""
    return os.getenv("USER", "default")


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run shell command with proper error handling."""
    try:
        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=check,
        )
        
        if capture_output and result.stdout:
            console.print(result.stdout)
        
        return result
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed: {' '.join(cmd)}[/red]")
        if e.stdout:
            console.print(f"[red]stdout: {e.stdout}[/red]")
        if e.stderr:
            console.print(f"[red]stderr: {e.stderr}[/red]")
        raise


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    deps = {
        "docker": shutil.which("docker") is not None,
        "kubectl": shutil.which("kubectl") is not None,
        "k3d": shutil.which("k3d") is not None,
        "git": shutil.which("git") is not None,
    }
    
    return deps


def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    if not model_name:
        return False
    
    # Check if it's alphanumeric with hyphens and underscores
    return model_name.replace("-", "").replace("_", "").isalnum()


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Failed to load YAML file {file_path}: {e}[/red]")
        raise


def save_yaml_file(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to YAML file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, indent=2)
    except Exception as e:
        console.print(f"[red]Failed to save YAML file {file_path}: {e}[/red]")
        raise


def print_table(data: List[Dict[str, Any]], title: str = "") -> None:
    """Print data as a formatted table."""
    if not data:
        console.print("[yellow]No data to display[/yellow]")
        return
    
    table = Table(title=title)
    
    # Add columns
    if data:
        for key in data[0].keys():
            table.add_column(key.title(), style="cyan")
        
        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row.values()])
    
    console.print(table)


def get_kubectl_context() -> Optional[str]:
    """Get current kubectl context."""
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def check_k8s_connection() -> bool:
    """Check if kubectl can connect to cluster."""
    try:
        subprocess.run(
            ["kubectl", "get", "nodes"],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False