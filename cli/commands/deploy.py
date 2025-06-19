"""Deploy command for Kubernetes deployments."""

from pathlib import Path
from typing import Optional, Dict, Any

import typer
from rich.console import Console

from cli.utils import (
    get_project_root, validate_model_name, run_command, 
    get_user_prefix, check_k8s_connection, save_yaml_file, get_git_sha
)

app = typer.Typer()
console = Console()


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model service to deploy"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Resource prefix"),
    registry: str = typer.Option("gcr.io/model-zoo", "--registry", "-r", help="Docker registry"),
    gpus: Optional[str] = typer.Option("nvidia-l4", "--gpus", help="GPU node selector value"),
    replicas: int = typer.Option(1, "--replicas", help="Number of replicas"),
    gcs_path: Optional[str] = typer.Option(None, "--gcs-path", help="GCS path for model weights"),
    node_selector: Optional[str] = typer.Option(None, "--node-selector", help="Node selector (key=value)"),
) -> None:
    """Deploy model to Kubernetes."""
    
    # Validate model name
    if not validate_model_name(model_name):
        console.print(f"[red]Invalid model name: {model_name}[/red]")
        raise typer.Exit(1)
    
    # Check Kubernetes connection
    if not check_k8s_connection():
        console.print("[red]Cannot connect to Kubernetes cluster[/red]")
        console.print("[yellow]Make sure kubectl is configured and cluster is accessible[/yellow]")
        raise typer.Exit(1)
    
    # Get defaults
    if prefix is None:
        prefix = get_user_prefix()
    
    if namespace is None:
        namespace = f"{prefix}-zoo"
    
    if gcs_path is None:
        sha = get_git_sha()
        gcs_path = f"gs://model-zoo/{prefix}/{model_name}/{sha}/"
    
    project_root = get_project_root()
    service_dir = project_root / "services" / model_name.lower()
    
    # Check if service exists
    if not service_dir.exists():
        console.print(f"[red]Service not found: {service_dir}[/red]")
        console.print(f"[yellow]Run 'zoo init {model_name}' first[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Deploying {model_name} to Kubernetes[/blue]")
    console.print(f"[dim]Namespace: {namespace}[/dim]")
    console.print(f"[dim]Registry: {registry}[/dim]")
    console.print(f"[dim]GCS Path: {gcs_path}[/dim]")
    
    # Parse node selector
    node_selector_dict = {}
    if node_selector:
        try:
            key, value = node_selector.split("=", 1)
            node_selector_dict[key] = value
        except ValueError:
            console.print(f"[red]Invalid node selector format: {node_selector}[/red]")
            console.print("[yellow]Use format: key=value[/yellow]")
            raise typer.Exit(1)
    elif gpus:
        node_selector_dict["accelerator"] = gpus
    
    # Generate Kubernetes manifests
    sha = get_git_sha()
    image_tag = f"{registry}/{prefix}-{model_name}:gpu-{sha}"
    
    try:
        # Create namespace if needed
        _create_namespace(namespace)
        
        # Generate and apply manifests
        manifests = _generate_manifests(
            model_name=model_name,
            namespace=namespace,
            prefix=prefix,
            image_tag=image_tag,
            replicas=replicas,
            gcs_path=gcs_path,
            node_selector=node_selector_dict,
        )
        
        # Apply manifests
        _apply_manifests(manifests, namespace)
        
        console.print(f"[green]✓ Successfully deployed {model_name}![/green]")
        console.print(f"[yellow]Check status: kubectl get pods -n {namespace}[/yellow]")
        console.print(f"[yellow]View logs: zoo logs {model_name}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Deployment failed: {e}[/red]")
        raise typer.Exit(1)


def _create_namespace(namespace: str) -> None:
    """Create Kubernetes namespace if it doesn't exist."""
    try:
        run_command(["kubectl", "create", "namespace", namespace], check=False)
        console.print(f"[green]✓ Created namespace: {namespace}[/green]")
    except Exception:
        # Namespace might already exist, which is fine
        pass


def _generate_manifests(
    model_name: str,
    namespace: str,
    prefix: str,
    image_tag: str,
    replicas: int,
    gcs_path: str,
    node_selector: Dict[str, str],
) -> Dict[str, Any]:
    """Generate Kubernetes manifests."""
    
    labels = {
        "app": f"{prefix}-{model_name}",
        "model": model_name,
        "prefix": prefix,
    }
    
    # Deployment
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{prefix}-{model_name}",
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": model_name,
                            "image": image_tag,
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "MODEL_GCS_PATH", "value": gcs_path},
                                {"name": "MODEL_NAME", "value": model_name},
                                {"name": "DEVICE", "value": "cuda:0"},
                                {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "4Gi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": "8Gi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 60,
                                "periodSeconds": 30,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                        }
                    ],
                    "nodeSelector": node_selector,
                }
            }
        }
    }
    
    # Service
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{prefix}-{model_name}",
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "selector": labels,
            "ports": [
                {
                    "name": "http",
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP",
                }
            ],
            "type": "ClusterIP",
        }
    }
    
    # HPA
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": f"{prefix}-{model_name}",
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": f"{prefix}-{model_name}",
            },
            "minReplicas": 1,
            "maxReplicas": 10,
            "metrics": [
                {
                    "type": "Pods",
                    "pods": {
                        "metric": {"name": "queue_depth"},
                        "target": {"type": "AverageValue", "averageValue": "1"},
                    },
                }
            ],
        }
    }
    
    return {
        "deployment": deployment,
        "service": service,
        "hpa": hpa,
    }


def _apply_manifests(manifests: Dict[str, Any], namespace: str) -> None:
    """Apply Kubernetes manifests."""
    
    # Create temporary directory for manifests
    temp_dir = Path("/tmp/zoo-deploy")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save manifests to files
        for name, manifest in manifests.items():
            manifest_file = temp_dir / f"{name}.yaml"
            save_yaml_file(manifest, manifest_file)
        
        # Apply manifests
        for manifest_file in temp_dir.glob("*.yaml"):
            run_command(["kubectl", "apply", "-f", str(manifest_file)])
    
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)