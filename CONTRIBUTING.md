# Contributing to FastAPI Inference Menagerie

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Report issues you've encountered
- ✨ **Feature Requests**: Suggest new features or improvements
- 🔧 **Bug Fixes**: Fix identified issues
- 📖 **Documentation**: Improve or add documentation
- 🧪 **Tests**: Add or improve test coverage
- 🚀 **New Models**: Add support for new ML models
- ⚡ **Performance**: Optimize existing code

### Before Contributing

1. **Check existing issues** to avoid duplicate work
2. **Discuss major changes** by creating an issue first
3. **Review the codebase** to understand the architecture
4. **Read this guide** thoroughly

## Development Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- kubectl and k3d
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/user/fastapi-inference-menagerie-claude.git
cd fastapi-inference-menagerie-claude

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Create local k3d cluster
./scripts/k3d/create.sh

# Verify setup
pytest tests/unit/ -v
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/unit/ -v
./scripts/dev/run_local.sh clip

# Run linting and formatting
black .
ruff check .
mypy .

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

## Contributing Process

### 1. Issue Creation

For bugs:
- Use the bug report template
- Include reproduction steps
- Provide environment details
- Attach relevant logs

For features:
- Use the feature request template
- Explain the problem you're solving
- Describe your proposed solution
- Consider alternatives

### 2. Development

1. **Fork and Clone**: Fork the repo and clone your fork
2. **Branch**: Create a feature branch from `main`
3. **Develop**: Make your changes following our standards
4. **Test**: Ensure all tests pass and add new tests
5. **Document**: Update documentation as needed

### 3. Pull Request

1. **Create PR**: Use our PR template
2. **Description**: Clearly describe your changes
3. **Link Issues**: Reference related issues
4. **Request Review**: Tag relevant maintainers
5. **Address Feedback**: Respond to review comments

### 4. Review Process

- All PRs require at least one review
- Automated checks must pass
- Maintainers will provide feedback
- Address comments and update PR
- Once approved, maintainers will merge

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use black for formatting
black .

# Use ruff for linting
ruff check .

# Use mypy for type checking
mypy .
```

### Code Organization

```
services/
├── common/          # Shared utilities
│   ├── adapter.py   # Base model adapter
│   ├── app.py       # FastAPI core
│   └── utils.py     # Utility functions
├── clip/            # CLIP model service
│   ├── adapter.py   # CLIP-specific adapter
│   └── app.py       # CLIP FastAPI app
└── new_model/       # Your new model
    ├── adapter.py   # Model adapter
    └── app.py       # FastAPI app
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ModelAdapter`)
- **Functions/Variables**: `snake_case` (e.g., `load_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Files**: `snake_case.py` (e.g., `model_adapter.py`)

### Type Hints

Always use type hints:

```python
from typing import Dict, Any, Optional, List

async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run model inference."""
    pass

def process_images(images: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Process list of images."""
    pass
```

### Error Handling

Use specific exceptions and proper logging:

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = await model.predict(data)
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Documentation Strings

Use Google-style docstrings:

```python
def compute_similarity(texts: List[str], images: List[str]) -> Dict[str, Any]:
    """Compute text-image similarity matrix.
    
    Args:
        texts: List of text descriptions
        images: List of base64 encoded images
        
    Returns:
        Dictionary containing similarity matrix and metadata
        
    Raises:
        ValueError: If inputs are empty or invalid
        RuntimeError: If model is not loaded
    """
    pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_adapter.py
│   └── test_utils.py
├── e2e/               # End-to-end tests
│   └── test_deployment.py
└── performance/       # Performance tests
    └── test_load.py
```

### Writing Tests

#### Unit Tests

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.unit
class TestModelAdapter:
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ModelAdapter("gs://bucket/model", "cpu")
        assert adapter.device == "cpu"
        assert not adapter.is_loaded
    
    @pytest.mark.asyncio
    async def test_predict_success(self):
        """Test successful prediction."""
        adapter = MockModelAdapter()
        result = await adapter.predict({"test": "data"})
        assert "result" in result
```

#### Integration Tests

```python
@pytest.mark.e2e
class TestCLIPService:
    def test_text_encoding_endpoint(self, test_client):
        """Test text encoding API endpoint."""
        response = test_client.post(
            "/v1/clip/encode/text",
            json={"texts": ["test text"]}
        )
        assert response.status_code == 200
        assert "result" in response.json()
```

#### Performance Tests

```python
@pytest.mark.performance
class TestPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        # Test implementation
        pass
```

### Test Coverage

- Aim for >90% test coverage
- All new features must include tests
- Bug fixes should include regression tests

```bash
# Run tests with coverage
pytest --cov=services --cov=cli --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation

### Types of Documentation

1. **Code Comments**: For complex logic
2. **Docstrings**: For all public functions/classes
3. **README**: Project overview and quick start
4. **API Docs**: Automatically generated from FastAPI
5. **User Guides**: Step-by-step tutorials
6. **Developer Docs**: Architecture and design decisions

### Documentation Standards

- Write for your audience (users vs developers)
- Include code examples
- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include diagrams for complex concepts

### Updating Documentation

When you make changes:

1. **Update docstrings** if function signatures change
2. **Update README** if user-facing features change
3. **Add examples** for new features
4. **Update CLI help** if commands change

## Adding New Models

### Model Adapter Requirements

1. **Inherit from ModelAdapter**:
```python
class MyModelAdapter(ModelAdapter):
    async def load_model(self) -> None:
        # Load your model
        pass
    
    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Implement inference
        pass
```

2. **Handle GCS weights**:
```python
async def load_model(self) -> None:
    weights_path = await self.download_weights()
    if weights_path:
        # Load from local weights
        self.model = load_from_path(weights_path)
    else:
        # Fallback to HuggingFace Hub
        self.model = load_from_hub("model-name")
```

3. **Add FastAPI routes**:
```python
@app.post("/v1/mymodel/predict")
async def predict(request: PredictRequest) -> Dict[str, Any]:
    result = await adapter.predict(request.dict())
    return {"success": True, "result": result}
```

### Model Integration Checklist

- [ ] Model adapter implements required methods
- [ ] FastAPI app with proper error handling
- [ ] Input/output validation with Pydantic
- [ ] Unit tests for adapter
- [ ] Integration tests for API endpoints
- [ ] Documentation and examples
- [ ] Docker build support
- [ ] CLI integration

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Create release PR** with version updates
4. **Merge PR** after review
5. **Tag release**: `git tag v1.0.0`
6. **Push tag**: `git push origin v1.0.0`
7. **GitHub Actions** will handle the rest

### Release Notes

Include in release notes:
- New features
- Bug fixes
- Breaking changes
- Migration instructions
- Performance improvements

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Code Review**: For feedback on implementations

## Recognition

Contributors will be:
- Listed in release notes
- Added to CONTRIBUTORS.md
- Recognized in community discussions

Thank you for contributing to the FastAPI Inference Menagerie! 🦁