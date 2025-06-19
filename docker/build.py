#!/usr/bin/env python3
"""Docker build script with Jinja2 templating."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import jinja2
from rich.console import Console

console = Console()


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


def render_dockerfile(
    template_path: Path,
    output_path: Path,
    context: Dict[str, Any],
) -> None:
    """Render Jinja2 Dockerfile template."""
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)
    
    rendered = template.render(**context)
    
    with open(output_path, "w") as f:
        f.write(rendered)
    
    console.print(f"[green]Rendered Dockerfile: {output_path}[/green]")


def build_image(
    dockerfile_path: Path,
    image_tag: str,
    context_path: Path = Path("."),
    build_args: Optional[Dict[str, str]] = None,
) -> None:
    """Build Docker image."""
    cmd = [
        "docker", "build",
        "-f", str(dockerfile_path),
        "-t", image_tag,
    ]
    
    if build_args:
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])
    
    cmd.append(str(context_path))
    
    console.print(f"[blue]Building image: {image_tag}[/blue]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print("[red]Build failed![/red]")
        sys.exit(1)
    
    console.print(f"[green]Successfully built: {image_tag}[/green]")


def push_image(image_tag: str) -> None:
    """Push Docker image to registry."""
    cmd = ["docker", "push", image_tag]
    
    console.print(f"[blue]Pushing image: {image_tag}[/blue]")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print("[red]Push failed![/red]")
        sys.exit(1)
    
    console.print(f"[green]Successfully pushed: {image_tag}[/green]")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build and push Docker images")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("--gpu", action="store_true", help="Build GPU image")
    parser.add_argument("--cpu", action="store_true", help="Build CPU image")
    parser.add_argument("--registry", default="gcr.io/model-zoo", help="Docker registry")
    parser.add_argument("--prefix", help="Resource prefix (defaults to $USER)")
    parser.add_argument("--base-tag", help="Base image tag")
    parser.add_argument("--pytorch-version", help="PyTorch version")
    parser.add_argument("--extra-depends", help="Extra pip dependencies")
    parser.add_argument("--push", action="store_true", help="Push after build")
    parser.add_argument("--output-dir", type=Path, default=Path("build"), help="Output directory")
    
    args = parser.parse_args()
    
    # Defaults
    if not args.gpu and not args.cpu:
        args.gpu = True  # Default to GPU
    
    if args.prefix is None:
        args.prefix = os.getenv("USER", "default")
    
    # Get git SHA
    sha = get_git_sha()
    
    # Build context
    docker_dir = Path(__file__).parent
    project_root = docker_dir.parent
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Template context
    context = {
        "MODEL_NAME": args.model_name.lower(),
        "BASE_TAG": args.base_tag,
        "PYTORCH_VERSION": args.pytorch_version,
        "EXTRA_DEPENDS": args.extra_depends,
        "USER_ID": os.getuid(),
        "GROUP_ID": os.getgid(),
    }
    
    # Build images
    if args.gpu:
        template_path = docker_dir / "Dockerfile.gpu.j2"
        dockerfile_path = args.output_dir / f"Dockerfile.{args.model_name}.gpu"
        image_tag = f"{args.registry}/{args.prefix}-{args.model_name}:gpu-{sha}"
        
        render_dockerfile(template_path, dockerfile_path, context)
        build_image(dockerfile_path, image_tag, project_root)
        
        if args.push:
            push_image(image_tag)
    
    if args.cpu:
        template_path = docker_dir / "Dockerfile.cpu.j2"
        dockerfile_path = args.output_dir / f"Dockerfile.{args.model_name}.cpu"
        image_tag = f"{args.registry}/{args.prefix}-{args.model_name}:cpu-{sha}"
        
        render_dockerfile(template_path, dockerfile_path, context)
        build_image(dockerfile_path, image_tag, project_root)
        
        if args.push:
            push_image(image_tag)


if __name__ == "__main__":
    main()