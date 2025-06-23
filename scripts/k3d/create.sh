#!/bin/bash
set -e

# k3d cluster creation script for local development

CLUSTER_NAME=${1:-"model-zoo"}
GPU_SUPPORT=${2:-"false"}
AGENTS=${3:-"1"}
PORT=${4:-"8080"}

echo "Creating k3d cluster: $CLUSTER_NAME"
echo "GPU Support: $GPU_SUPPORT"
echo "Agents: $AGENTS"
echo "Port mapping: localhost:$PORT -> cluster:80"

# Check if k3d is installed
if ! command -v k3d &> /dev/null; then
    echo "Error: k3d is not installed"
    echo "Install with: curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash"
    exit 1
fi

# Check if cluster already exists
if k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "Cluster $CLUSTER_NAME already exists"
    echo "Delete with: k3d cluster delete $CLUSTER_NAME"
    exit 1
fi

# Build k3d command
K3D_CMD="k3d cluster create $CLUSTER_NAME"
K3D_CMD="$K3D_CMD --agents $AGENTS"
K3D_CMD="$K3D_CMD --port $PORT:80@loadbalancer"
K3D_CMD="$K3D_CMD --port 6443:6443@server:0"

# Add GPU support if requested
if [ "$GPU_SUPPORT" = "true" ]; then
    echo "Adding GPU support..."
    # Note: This requires nvidia-docker2 and nvidia-container-toolkit
    K3D_CMD="$K3D_CMD --gpus all"
    
    # Check if nvidia-docker is available
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo "Warning: GPU support requested but nvidia-docker doesn't seem to work"
        echo "You may need to install nvidia-docker2 and nvidia-container-toolkit"
    fi
fi

# Add registry for local development
K3D_CMD="$K3D_CMD --registry-create model-zoo-registry:7070"

# Create cluster
echo "Running: $K3D_CMD"
eval $K3D_CMD

# Wait for cluster to be ready
echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

# Install NVIDIA GPU Operator if GPU support is enabled
if [ "$GPU_SUPPORT" = "true" ]; then
    echo "Installing NVIDIA GPU Operator..."
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
    helm repo update
    
    helm install --wait \
        --generate-name \
        -n gpu-operator --create-namespace \
        nvidia/gpu-operator \
        --set driver.enabled=false
fi

# Create model-zoo namespace
echo "Creating model-zoo namespace..."
kubectl create namespace model-zoo || true

# Set default namespace
kubectl config set-context --current --namespace=model-zoo

echo ""
echo "✅ k3d cluster '$CLUSTER_NAME' created successfully!"
echo ""
echo "Next steps:"
echo "1. kubectl get nodes"
echo "2. kubectl get pods -n kube-system"
if [ "$GPU_SUPPORT" = "true" ]; then
    echo "3. kubectl get pods -n gpu-operator"
fi
echo "4. zoo deploy <model-name>"
echo ""
echo "Access your services at: http://localhost:$PORT"
echo ""
echo "To delete the cluster:"
echo "k3d cluster delete $CLUSTER_NAME"
