#!/bin/bash
set -e

# Installation script for FastAPI Inference Menagerie

echo "🦁 Installing FastAPI Inference Menagerie..."

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    echo "✅ Python $PYTHON_VERSION is compatible (>= $REQUIRED_VERSION)"
else
    echo "❌ Python $PYTHON_VERSION is not compatible. Please install Python $REQUIRED_VERSION or higher."
    exit 1
fi

# Install with pip (recommended)
echo "📦 Installing package in editable mode..."
pip install -e .

echo ""
echo "🎉 Installation completed!"
echo ""
echo "Next steps:"
echo "  1. Create a k3d cluster: ./scripts/k3d/create.sh"
echo "  2. Initialize a model: zoo init my-model"
echo "  3. Start development: ./scripts/dev/run_local.sh my-model"
echo ""