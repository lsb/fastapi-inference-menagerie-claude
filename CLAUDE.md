# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a model zoo for deploying cutting-edge ML research models (CLIP, Grounding DINO + SAM2, Qwen 2.5 VL) to Kubernetes using FastAPI. The system is designed to run on GCP Kubernetes and locally via k3d.

## Key Commands

### Development
```bash
# Initialize a new model service
zoo init <MODEL_NAME>

# Build Docker image for a model
zoo build <MODEL_NAME> --cuda 12.4 --registry gcr.io/foo

# Deploy model to Kubernetes
zoo deploy <MODEL_NAME> --gpus nvidia-l4

# View logs
zoo logs <MODEL_NAME>

# Delete deployment
zoo delete <MODEL_NAME>

# Run unit tests
pytest -m unit

# Run e2e tests
pytest -m e2e

# Local development with hot reload
scripts/dev/run_local.sh

# Create local k3d cluster
scripts/k3d/create.sh
```

### Linting and Type Checking
```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy .
```

## Architecture

### Directory Structure
- `cli/` - Typer-based CLI for managing model deployments
- `services/` - FastAPI applications (one directory per model)
  - `services/common/` - Shared FastAPI core and ModelAdapter base class
- `charts/` - Kustomize bases and overlays for Kubernetes manifests
- `docker/` - Dockerfile templates (Jinja2) for GPU/CPU builds
- `scripts/` - Helper scripts for k3d setup and local development
- `tests/` - Pytest unit and e2e tests

### Key Components

1. **ModelAdapter Base Class**: All models inherit from `services/common/ModelAdapter` which provides:
   - GCS weight loading interface
   - Standard predict/stream methods
   - Device configuration

2. **FastAPI Service Pattern**: Each model service follows:
   - Prometheus metrics exposed at `/metrics`
   - Queue depth tracking middleware
   - SSE streaming support for generative models

3. **Kubernetes Deployment**:
   - One GPU per Pod via nodeSelector
   - HPA based on custom queue_depth metric
   - GCS access via workload identity (service account annotation)
   - Weights at `gs://model-zoo/<user>/<model>/<SHA>/`

4. **CLI Architecture**:
   - Uses Typer for command parsing
   - Generates Kustomize overlays dynamically
   - Resource prefix: `$USER-` or `--prefix` flag
   - Default namespace: `$PREFIX-zoo`

## Model-Specific Endpoints

- **CLIP**: `/v1/clip/encode` (JSON in/out)
- **Grounding DINO + SAM2**: `/v1/ground/segment` (returns boxes + masks base64)
- **Qwen 2.5 VL**: `/v1/vqa/ask` (SSE streaming)

## Configuration Details

- **Base Images**: NVIDIA PyTorch containers (`nvcr.io/nvidia/pytorch`)
- **GPU Scheduling**: One GPU per Pod, CUDA_VISIBLE_DEVICES=0
- **Scaling**: HPA triggers when queue_depth > 0
- **Logging**: Structured logs to stdout with format: `[timestamp] level=model INFO message="duration_ms=42 queue_depth=3"`