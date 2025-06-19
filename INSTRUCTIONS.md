> I’m trying to make a model zoo for some machine learning models. I want to run some CLIP, Grounding DINO + SAM2, and Qwen 2.5 VL for some visual question answering; implementations in Huggingface are fine. I want to use FastAPI to deploy arbitrary cutting edge ML research models to Kubernetes. I’m going to want to specify something like what GPU to use with nodeSelectors and annotations but otherwise I’m fine with defaults. We will be storing fine-tunes and we have k8s set up so that with the correct metadata annotation for GCP service account and with the right service account name we can automatically access a particular GCS bucket where the fine tunes will live. We want a command line interface for a team to use to each be able to develop and deploy models independently. We will run in GCP on Kubernetes and locally (via minikube/k3s/k3d/whatever makes sense).

### Multi-Phase Prompt for a Coding-Agent

*Goal: generate a production-ready repo that lets any teammate develop, package, and deploy PyTorch models (CLIP, Grounding-DINO + SAM 2, Qwen-2.5 VL) to Kubernetes—both in GCP and locally via k3d—using FastAPI for serving and a Typer-based CLI for automation.*

---

## Global Inputs (already decided)

| Area                     | Decision                                                              |
| ------------------------ | --------------------------------------------------------------------- |
| **Framework**            | PyTorch (JAX optional later)                                          |
| **Runtime images**       | NVIDIA PyTorch base images; CUDA tag is a config knob                 |
| **GPU scheduling**       | One GPU per Pod, nodeSelector key/value provided in config            |
| **Namespaces**           | Default shared namespace, overrideable per deploy                     |
| **Weights**              | `gs://model-zoo/<user>/<model>/<SHA>/...`, streamed at runtime        |
| **CLI language / verbs** | Python + Typer; `init, build, deploy, update, rollback, logs, delete` |
| **Local K8s**            | k3d (GPU passthrough optional)                                        |
| **Scaling**              | One-model-per-service; HPA on custom queue-depth metric               |
| **Observability**        | Logs & Prometheus metrics to stdout                                   |
| **CI/CD**                | GitHub Actions, unit + e2e tests (no scans or signing)                |
| **Resource prefix**      | `$USER-` or `--prefix` flag                                           |

---

## Phase 0 – Repository Scaffold

1. **Create repo** `model-zoo-fastapi`.
2. Top-level dirs:

   ```
   .github/workflows/      # CI
   cli/                    # Typer source
   services/               # FastAPI apps (one dir per model)
   charts/                 # kustomize bases & overlays
   docker/                 # Dockerfile templates
   scripts/                # helper bash/python
   tests/                  # pytest + e2e
   ```
3. Add **pyproject.toml** (hatchling) with black, ruff, mypy configured.

---

## Phase 1 – Docker Image Templates

*Goal: produce fully parameterised Dockerfiles for GPU and CPU builds.*

1. Under `docker/`, create `Dockerfile.gpu.j2` with variables: `BASE_TAG` (e.g. `nvcr.io/nvidia/pytorch:24.03-py3`), `PYTORCH_VERSION`, `EXTRA_DEPENDS`, `USER_ID`, `GROUP_ID`.
2. Inject build args at `cli build`.
3. Provide script `docker/build.py` to render Jinja template, build, and push to `$REGISTRY/$PREFIX-$MODEL:$SHA`.

---

## Phase 2 – Common FastAPI Service Core

1. Provide `services/common/app.py` exposing:

   ```python
   from fastapi import FastAPI, Request
   from sse_starlette.sse import EventSourceResponse
   app = FastAPI(title="Model Zoo Service")

   @app.middleware("http")
   async def add_queue_depth_metric(request: Request, call_next):
       # increment/decrement atomic counter
   ```
2. Include **abstract base class** `ModelAdapter` with:

   ```python
   class ModelAdapter:
       def __init__(self, gcs_path: str, device: str): ...
       async def predict(self, payload: dict) -> dict: ...
       async def stream(self, payload: dict):  # yields str/bytes
   ```
3. Add Prometheus client that writes metrics to stdout (`text_exposition_format`) every 5 s.

---

## Phase 3 – Model-Specific Adapters

### 3.1 CLIP

* Load HF weights from env `MODEL_GCS_PATH`.
* Expose `/v1/clip/encode` (JSON in, JSON out).

### 3.2 Grounding-DINO + SAM 2

* Two-stage pipeline inside `/v1/ground/segment`.
* Return boxes + masks base64.

### 3.3 Qwen-2.5 VL

* Endpoint `/v1/vqa/ask` with SSE streaming of tokens.

Unit tests: CPU-only dummy forward that verifies JSON schema.

---

## Phase 4 – Typer CLI

```
$ zoo init CLIP            # copies template into services/clip
$ zoo build CLIP --cuda 12.4 --registry gcr.io/foo
$ zoo deploy CLIP --gpus nvidia-l4
$ zoo logs CLIP
$ zoo delete CLIP
```

Key design points:

| Flag                      | Behaviour                 |
| ------------------------- | ------------------------- |
| `--node-selector KEY=VAL` | overrides config file     |
| `--namespace`             | defaults to `$PREFIX-zoo` |
| `--prefix`                | defaults to `$USER` env   |

CLI generates rendered kustomize overlay → `kubectl apply -k`.

---

## Phase 5 – Kubernetes Manifests (Kustomize)

1. **Base**: Deployment, Service, HPAv2, ServiceAccount.
2. Patches inserted by CLI:

   * container image
   * resource requests (`gpu: 1`, `cpu: 2`, `memory: 4Gi`)
   * `nodeSelector`
   * env vars: `MODEL_GCS_PATH`, `CUDA_VISIBLE_DEVICES=0`
3. HPA template uses `queue_depth` metric from `/metrics`.

---

## Phase 6 – GCS Weight Loader

* In each adapter’s `__init__`, shell out to `gsutil cp` → `/models/weights` (or use `gcsfs` with stream download).
* Provide optional **local cache** directory `/var/cache/zoo`.

---

## Phase 7 – Local Dev via k3d

1. `scripts/k3d/create.sh` spins up 1-node cluster (`--agents 1 --gpu 1` when available).
2. `scripts/dev/run_local.sh` starts FastAPI with `uvicorn --reload` on CPU.

---

## Phase 8 – Observability

1. Stdout formatting:

   ```
   [timestamp] level=model INFO message="duration_ms=42 queue_depth=3"
   ```
2. Prometheus exporter already handled in Phase 2.

---

## Phase 9 – Testing

1. **Unit**: `pytest -m unit`, run on GitHub Actions (`ubuntu-latest`).
2. **e2e**: Spin up k3d, deploy dummy CLIP, hit `/healthz`, and run a sample request.
3. Use `pytest-asyncio` for streaming tests.

---

## Phase 10 – GitHub Actions

* **build.yml** — trigger on PR; run lint, unit tests.
* **deploy.yml** — on main, build Docker, push, run e2e, then `kubectl apply` to staging (optional).

---

## Phase 11 – Docs & Examples

1. `README.md` high-level overview + quickstart.
2. `docs/` folder with:

   * `design.md`
   * `cli.md`
   * `models/{clip,dino-sam2,qwen-vl}.md`
3. Add **OpenAPI** json generation in CI and publish to GitHub Pages.

---

### Acceptance Criteria Checklist

1. `zoo init/build/deploy` works for each reference model.
2. A PR with new model adapter passes lint, unit, and e2e by default.
3. Deploy to GKE with `--node-selector accelerator=nvidia-l4` spins up Pod that pulls weights from GCS and answers inference in < 2 × baseline latency.
4. HPA scales replicas 0→1 when `queue_depth > 0`.
5. Logs and metrics visible in `kubectl logs`.

---

**Hand this entire spec to the coding agent.**
It defines every deliverable, directory, template variable, and contract it needs to implement, so the agent can proceed phase-by-phase without ambiguity.
