# Deployment Examples

This document provides comprehensive examples for deploying models using the Model Zoo CLI.

## Table of Contents

- [Local Development](#local-development)
- [Staging Deployment](#staging-deployment)
- [Production Deployment](#production-deployment)
- [Multi-Model Deployment](#multi-model-deployment)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## Local Development

### Setting up k3d Cluster

```bash
# Create local k3d cluster with GPU support
./scripts/k3d/create.sh model-zoo true 2 8080

# Verify cluster is running
kubectl get nodes
kubectl get pods -A
```

### Developing a New Model

```bash
# Initialize new model service
zoo init my-awesome-model --template basic

# Edit the adapter and app files
vim services/my-awesome-model/adapter.py
vim services/my-awesome-model/app.py

# Test locally with hot reload
./scripts/dev/run_local.sh my-awesome-model

# Test API endpoints
./scripts/dev/test_api.sh http://localhost:8000 my-awesome-model
```

### Quick Local Deployment

```bash
# Build and deploy to local k3d
zoo build my-awesome-model --registry localhost:5000
zoo deploy my-awesome-model --namespace default

# Check status
zoo logs my-awesome-model --follow
kubectl get pods
```

## Staging Deployment

### Environment Setup

```bash
# Configure GCP credentials
gcloud auth login
gcloud config set project my-gcp-project

# Get cluster credentials
gcloud container clusters get-credentials staging-cluster \
  --zone us-central1-a
```

### Deploy to Staging

```bash
# Build and push to container registry
zoo build clip \
  --registry gcr.io/my-project \
  --gpu \
  --push

# Deploy to staging environment
zoo deploy clip \
  --namespace staging-zoo \
  --registry gcr.io/my-project \
  --gcs-path gs://my-bucket/staging/clip/v1.0/ \
  --gpus nvidia-t4 \
  --replicas 2

# Monitor deployment
kubectl get pods -n staging-zoo
zoo logs clip --namespace staging-zoo --follow
```

### Staging Testing

```bash
# Port forward for testing
kubectl port-forward -n staging-zoo service/staging-clip 8080:80

# Run integration tests
pytest tests/e2e/ --staging-url http://localhost:8080

# Load testing
./scripts/load_test.sh http://localhost:8080 100 60
```

## Production Deployment

### Pre-deployment Checklist

- [ ] All tests pass in CI
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Rollback plan prepared

### Deploy to Production

```bash
# Tag release
git tag v1.2.0
git push origin v1.2.0

# This triggers automatic deployment via GitHub Actions
# Or deploy manually:

zoo deploy clip \
  --namespace production-zoo \
  --registry gcr.io/my-project \
  --gcs-path gs://my-bucket/production/clip/v1.2.0/ \
  --gpus nvidia-a100 \
  --replicas 5 \
  --node-selector environment=production,zone=us-central1-a

# Verify deployment
kubectl get deployments -n production-zoo
kubectl get hpa -n production-zoo
zoo logs clip --namespace production-zoo --tail 100
```

### Blue-Green Deployment

```bash
# Deploy new version to "green" environment
zoo deploy clip \
  --namespace production-zoo-green \
  --gcs-path gs://my-bucket/production/clip/v1.3.0/ \
  --replicas 5

# Test green environment
./scripts/smoke_test.sh production-zoo-green

# Switch traffic (update load balancer)
kubectl patch service clip-service -n production-zoo \
  -p '{"spec":{"selector":{"app":"production-green-clip"}}}'

# Monitor for issues
kubectl get pods -n production-zoo-green -w

# Cleanup old environment after verification
zoo delete clip --namespace production-zoo-blue --force
```

## Multi-Model Deployment

### Deploy All Models

```bash
# Deploy all models with consistent configuration
for model in clip grounding-sam qwen-vl; do
  echo "Deploying $model..."
  
  zoo build $model --gpu --push
  zoo deploy $model \
    --namespace prod-zoo \
    --gpus nvidia-a100 \
    --replicas 3 \
    --gcs-path gs://my-bucket/prod/$model/latest/
  
  # Wait for deployment
  kubectl wait --for=condition=available \
    deployment/prod-$model -n prod-zoo --timeout=600s
done

echo "All models deployed successfully!"
```

### Model-Specific Configuration

```bash
# CLIP with high throughput
zoo deploy clip \
  --replicas 10 \
  --gpus nvidia-t4 \
  --node-selector workload=inference

# Grounding SAM with powerful GPUs
zoo deploy grounding-sam \
  --replicas 3 \
  --gpus nvidia-a100 \
  --node-selector workload=computer-vision

# Qwen VL with streaming optimization
zoo deploy qwen-vl \
  --replicas 5 \
  --gpus nvidia-a100 \
  --node-selector workload=language-models
```

## Advanced Configuration

### Custom Resource Limits

```bash
# Deploy with custom resource limits
zoo deploy clip \
  --replicas 3 \
  --gpus nvidia-a100 \
  --cpu-request 4 \
  --cpu-limit 8 \
  --memory-request 8Gi \
  --memory-limit 16Gi
```

### Environment Variables

```bash
# Deploy with custom environment
zoo deploy clip \
  --env MODEL_CACHE_SIZE=1000 \
  --env LOG_LEVEL=DEBUG \
  --env BATCH_SIZE=32 \
  --env MAX_SEQUENCE_LENGTH=512
```

### Persistent Storage

```bash
# Deploy with persistent volume for caching
zoo deploy clip \
  --volume-size 100Gi \
  --volume-mount /var/cache/model \
  --storage-class fast-ssd
```

### Network Policies

```bash
# Apply network policies for security
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: clip-network-policy
  namespace: production-zoo
spec:
  podSelector:
    matchLabels:
      app: prod-clip
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for GCS
EOF
```

## Troubleshooting

### Common Issues

#### Pod Stuck in Pending

```bash
# Check node resources
kubectl describe nodes
kubectl top nodes

# Check pod events
kubectl describe pod <pod-name> -n <namespace>

# Check if GPU nodes are available
kubectl get nodes -l accelerator=nvidia-a100
```

#### Model Loading Errors

```bash
# Check GCS permissions
kubectl exec -it <pod-name> -n <namespace> -- \
  gsutil ls gs://my-bucket/models/

# Check model adapter logs
zoo logs clip --namespace <namespace> --tail 100

# Verify GCS path exists
gsutil ls -r gs://my-bucket/models/clip/
```

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n <namespace>

# Reduce batch size
kubectl set env deployment/<deployment-name> \
  BATCH_SIZE=16 -n <namespace>

# Add memory limits
kubectl patch deployment <deployment-name> -n <namespace> \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"model","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

#### Scaling Issues

```bash
# Check HPA status
kubectl get hpa -n <namespace>
kubectl describe hpa <hpa-name> -n <namespace>

# Check metrics server
kubectl top pods -n <namespace>

# Manual scaling
kubectl scale deployment <deployment-name> \
  --replicas 5 -n <namespace>
```

### Debugging Commands

```bash
# Get all resources for a model
kubectl get all -l app=prod-clip -n production-zoo

# Check resource usage
kubectl top pods -n production-zoo
kubectl top nodes

# View events
kubectl get events -n production-zoo --sort-by='.lastTimestamp'

# Debug networking
kubectl exec -it <pod-name> -n <namespace> -- netstat -tlnp
kubectl exec -it <pod-name> -n <namespace> -- nslookup kubernetes.default

# Check persistent volumes
kubectl get pv,pvc -n <namespace>
```

### Performance Optimization

```bash
# Enable resource quotas
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: model-quota
  namespace: production-zoo
spec:
  hard:
    requests.nvidia.com/gpu: "20"
    requests.cpu: "40"
    requests.memory: 200Gi
    limits.cpu: "80"
    limits.memory: 400Gi
EOF

# Set pod disruption budgets
kubectl apply -f - <<EOF
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: clip-pdb
  namespace: production-zoo
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: prod-clip
EOF

# Configure horizontal pod autoscaler
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: clip-hpa
  namespace: production-zoo
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prod-clip
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: queue_depth
      target:
        type: AverageValue
        averageValue: "2"
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
EOF
```

## Next Steps

- [Model Development Guide](models.md)
- [API Documentation](api.md)
- [Monitoring and Observability](monitoring.md)
- [Security Best Practices](security.md)