#!/bin/bash
set -e

# k3d cluster deletion script

CLUSTER_NAME=${1:-"model-zoo"}

echo "Deleting k3d cluster: $CLUSTER_NAME"

# Check if k3d is installed
if ! command -v k3d &> /dev/null; then
    echo "Error: k3d is not installed"
    exit 1
fi

# Check if cluster exists
if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "Cluster $CLUSTER_NAME does not exist"
    exit 1
fi

# Confirm deletion
read -p "Are you sure you want to delete cluster '$CLUSTER_NAME'? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deletion cancelled"
    exit 0
fi

# Delete cluster
echo "Deleting cluster..."
k3d cluster delete "$CLUSTER_NAME"

echo "✅ Cluster '$CLUSTER_NAME' deleted successfully!"