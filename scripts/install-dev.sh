#!/bin/bash
set -e

# Development installation script with all dependencies

echo "🦁 Installing FastAPI Inference Menagerie (Development Mode)..."

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    echo "✅ Python $PYTHON_VERSION is compatible (>= $REQUIRED_VERSION)"
else
    echo "❌ Python $PYTHON_VERSION is not compatible. Please install Python $REQUIRED_VERSION or higher."
    exit 1
fi

# Install with all development dependencies
echo "📦 Installing package with development dependencies..."
pip install -e ".[dev]"

echo ""
echo "🎉 Development installation completed!"
echo ""
echo "Available development tools:"
echo "  • pytest - Run tests"
echo "  • black - Code formatting"
echo "  • isort - Import sorting"
echo "  • flake8 - Linting"
echo "  • mypy - Type checking"
echo ""
echo "Next steps:"
echo "  1. Create a k3d cluster: ./scripts/k3d/create.sh"
echo "  2. Run tests: ./scripts/test/run-all.sh"
echo "  3. Initialize a model: zoo init my-model"
echo "  4. Start development: ./scripts/dev/run_local.sh my-model"
echo ""