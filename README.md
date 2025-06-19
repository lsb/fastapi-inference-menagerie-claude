# 🦁 FastAPI Inference Menagerie

> *A production-ready model zoo for deploying cutting-edge ML research models to Kubernetes*

[![CI](https://github.com/user/fastapi-inference-menagerie-claude/workflows/CI/badge.svg)](https://github.com/user/fastapi-inference-menagerie-claude/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Deploy CLIP, Grounding DINO + SAM2, Qwen 2.5 VL, and other ML models to Kubernetes with a single command. Supports both GCP and local k3d clusters with automatic scaling, observability, and CI/CD.

## ✨ Features

- 🚀 **One-Command Deployment**: `zoo deploy clip --gpus nvidia-l4`
- 🔄 **Auto-Scaling**: HPA based on queue depth metrics
- ☁️ **Cloud-Native**: GCS weight loading with workload identity
- 🐳 **Docker Templates**: GPU/CPU builds with Jinja2 parameterization
- 📊 **Observability**: Prometheus metrics, structured logging, health checks
- 🧪 **Local Development**: k3d cluster with hot-reload dev server
- 🔒 **Production-Ready**: Security scanning, automated testing, rollback support

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI (Typer)   │    │  Docker Build   │    │   Kubernetes    │
│                 │    │    (Jinja2)     │    │   (Kustomize)   │
│ • init          │───▶│ • GPU/CPU imgs  │───▶│ • Auto-scaling  │
│ • build         │    │ • Multi-stage   │    │ • Load balancing│
│ • deploy        │    │ • Optimized     │    │ • Health checks │
│ • logs/rollback │    └─────────────────┘    └─────────────────┘
└─────────────────┘                                    │
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Observability │    │  FastAPI Core   │    │   Model Zoo     │
│                 │    │                 │    │                 │
│ • Prometheus    │◀───│ • Common base   │◀───│ • CLIP          │
│ • Structured    │    │ • Metrics       │    │ • Grounding SAM │
│   logging       │    │ • Health checks │    │ • Qwen 2.5 VL   │
│ • Queue depth   │    │ • Streaming     │    │ • Custom models │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker
- kubectl
- k3d (for local development)

### Installation

```bash
# Clone the repository
git clone https://github.com/lsb/fastapi-inference-menagerie-claude.git
cd fastapi-inference-menagerie-claude

# Install with pip (recommended)
pip install -e .

# Or install with all development dependencies
pip install -e ".[dev]"
```

### Local Development

```bash
# Create k3d cluster
./scripts/k3d/create.sh

# Initialize a new model
zoo init my-model

# Start development server
./scripts/dev/run_local.sh my-model

# Test the API
./scripts/dev/test_api.sh http://localhost:8000 my-model
```

### Production Deployment

```bash
# Build and deploy CLIP to GKE
zoo build clip --push
zoo deploy clip --gpus nvidia-l4 --replicas 3

# Check status
zoo logs clip --follow

# Update deployment
zoo update clip --image-tag new-version

# Rollback if needed
zoo rollback clip
```

## 🤖 Supported Models

| Model | Endpoint | Description |
|-------|----------|-------------|
| **CLIP** | `/v1/clip/encode` | Text and image encoding with similarity |
| **Grounding DINO + SAM2** | `/v1/ground/segment` | Object detection and segmentation |
| **Qwen 2.5 VL** | `/v1/vqa/ask` | Visual question answering with streaming |

### CLIP Example

```python
import requests

# Text encoding
response = requests.post("http://localhost:8000/v1/clip/encode/text", 
    json={"texts": ["a cat", "a dog"]})
embeddings = response.json()["result"]["embeddings"]

# Image similarity
response = requests.post("http://localhost:8000/v1/clip/similarity",
    json={
        "texts": ["a cat"],
        "images": ["data:image/jpeg;base64,/9j/4AAQ..."]
    })
similarity = response.json()["result"]["similarity_matrix"]
```

### Grounding SAM Example

```python
# Object detection and segmentation
response = requests.post("http://localhost:8000/v1/ground/segment",
    json={
        "image": "data:image/jpeg;base64,/9j/4AAQ...",
        "text": "person walking",
        "confidence_threshold": 0.3
    })
boxes = response.json()["result"]["boxes"]
masks = response.json()["result"]["masks"]  # base64 encoded
```

### Qwen VL Example

```python
# Visual question answering
response = requests.post("http://localhost:8000/v1/vqa/ask",
    json={
        "image": "data:image/jpeg;base64,/9j/4AAQ...",
        "question": "What is in this image?",
        "stream": False
    })
answer = response.json()["result"]["answer"]

# Streaming response
import sseclient
response = requests.post("http://localhost:8000/v1/vqa/ask",
    json={"image": "...", "question": "...", "stream": True},
    stream=True)
client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data)  # Streaming tokens
```

## 🛠️ CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `zoo init <model>` | Initialize new model service | `zoo init bert-large` |
| `zoo build <model>` | Build Docker image | `zoo build clip --gpu --push` |
| `zoo deploy <model>` | Deploy to Kubernetes | `zoo deploy clip --replicas 3` |
| `zoo update <model>` | Update deployment | `zoo update clip --image-tag v2.0` |
| `zoo rollback <model>` | Rollback deployment | `zoo rollback clip --revision 2` |
| `zoo logs <model>` | View service logs | `zoo logs clip --follow` |
| `zoo delete <model>` | Delete deployment | `zoo delete clip --force` |

## 🔧 Configuration

### Environment Variables

```bash
export MODEL_GCS_PATH="gs://my-bucket/models/clip/v1.0/"
export MODEL_NAME="clip"
export DEVICE="cuda:0"
export LOG_LEVEL="INFO"
export CACHE_DIR="/var/cache/zoo"
```

### GCS Weight Loading

Models automatically download weights from GCS with local caching:

```bash
# Deploy with custom GCS path
zoo deploy clip --gcs-path gs://my-bucket/custom-clip/weights/
```

### GPU Scheduling

```bash
# Deploy with specific GPU type
zoo deploy clip --node-selector accelerator=nvidia-a100

# Deploy with custom node selector
zoo deploy clip --node-selector environment=production,zone=us-central1-a
```

## 📊 Monitoring & Observability

### Metrics

All services expose Prometheus metrics at `/metrics`:

- `model_requests_total` - Total requests by status
- `model_request_duration_seconds` - Request latency
- `model_queue_depth` - Current queue depth for HPA
- `model_load_time_seconds` - Model loading time

### Logging

Structured logs with consistent format:
```
[2024-01-15T10:30:45] level=model INFO message="Request processed duration_ms=150 queue_depth=2"
```

### Health Checks

- **Liveness**: `/health` - Model loaded and responding
- **Readiness**: `/health` - Ready to serve traffic
- **Startup**: Configurable delay for model loading

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests with k3d
pytest tests/e2e/ -v

# Performance tests
pytest tests/performance/ -v

# Run all tests
pytest -v
```

## 🚀 CI/CD

### GitHub Actions

- **CI**: Lint, test, security scan on every PR
- **CD**: Auto-deploy to staging on main branch
- **Release**: Tagged releases deploy to production
- **Cleanup**: Automated cleanup of old images and deployments

### Environments

1. **Development**: Local k3d cluster
2. **Staging**: GKE staging cluster (auto-deploy from main)
3. **Production**: GKE production cluster (manual approval required)

## 📚 Documentation

- [CLI Usage](docs/cli.md) - Detailed CLI documentation
- [Model Development](docs/models.md) - Adding new models
- [Deployment Guide](docs/deployment.md) - Kubernetes deployment
- [API Reference](docs/api.md) - Complete API documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research
- [SAM 2](https://github.com/facebookresearch/segment-anything-2) by Meta
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) by Alibaba Cloud
- [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramirez
- [Typer](https://typer.tiangolo.com/) by Sebastián Ramirez

---

Made with ❤️ for the ML community

> **Note**: This project is currently installed from source. PyPI package release is planned for future versions.
