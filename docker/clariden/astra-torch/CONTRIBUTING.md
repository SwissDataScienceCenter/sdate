# Contributing to ASTRA-Torch

We welcome contributions to ASTRA-Torch! This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended)
- ASTRA Toolbox with CUDA support

### Setting Up Your Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/astra-torch.git
   cd astra-torch
   ```

2. **Create a development environment:**
   ```bash
   # Using conda (recommended)
   conda create -n astra-torch-dev python=3.10
   conda activate astra-torch-dev
   conda install -c astra-toolbox astra-toolbox
   
   # Or using venv
   python -m venv astra-torch-dev
   source astra-torch-dev/bin/activate  # Linux/Mac
   # astra-torch-dev\Scripts\activate   # Windows
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev,notebooks]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Setup

### Project Structure

```
astra-torch/
├── astra_torch/           # Main package
│   ├── cbct.py           # CBCT reconstruction functions
│   ├── lamino.py         # Laminography functions
│   └── __init__.py       # Package initialization
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Example scripts and notebooks
├── .github/              # GitHub workflows
└── pyproject.toml        # Package configuration
```

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests locally:**
   ```bash
   pytest tests/
   ```

4. **Check code quality:**
   ```bash
   pre-commit run --all-files
   ```

## Code Style

We use several tools to maintain code quality:

### Code Formatting
- **Black**: Automatic code formatting
- **isort**: Import sorting

### Linting
- **flake8**: Style guide enforcement
- **mypy**: Static type checking

### Configuration
All tools are configured in:
- `.pre-commit-config.yaml`: Pre-commit hooks
- `.flake8`: Flake8 configuration  
- `mypy.ini`: MyPy configuration
- `pyproject.toml`: Black and isort settings

### Style Guidelines

1. **Follow PEP 8** for Python code style
2. **Use type hints** for all function signatures
3. **Write descriptive docstrings** using Google style:
   ```python
   def reconstruct_volume(projections: torch.Tensor, vectors: np.ndarray) -> torch.Tensor:
       """Reconstruct volume from projection data.
       
       Args:
           projections: Input projection data with shape (B, 1, N, H, W).
           vectors: ASTRA geometry vectors with shape (N, 12).
           
       Returns:
           Reconstructed volume with shape (B, 1, D, H, W).
           
       Raises:
           ValueError: If input shapes are incompatible.
       """
   ```

4. **Keep functions focused** and single-purpose
5. **Use meaningful variable names**
6. **Add comments for complex algorithms**

## Testing

### Test Structure

We use pytest for testing with the following structure:
- `tests/test_cbct.py`: CBCT function tests
- `tests/test_lamino.py`: Laminography function tests  
- `tests/conftest.py`: Test fixtures and configuration

### Writing Tests

1. **Create test files** following the `test_*.py` naming convention
2. **Use descriptive test names** that explain what is being tested
3. **Include both unit and integration tests**
4. **Test edge cases and error conditions**
5. **Use fixtures** for common test data

Example test:
```python
def test_cbct_reconstruction_shape(cbct_geometry, simple_projections):
    """Test that CBCT reconstruction returns correct shape."""
    volume = fdk_reconstruction_masked(
        simple_projections,
        cbct_geometry,
        volume_shape=(64, 64, 64)
    )
    assert volume.shape == (1, 1, 64, 64, 64)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_cbct.py

# Run with coverage
pytest --cov=astra_torch

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

## Documentation

### Building Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.

### Documentation Guidelines

1. **Keep docstrings up to date** with code changes
2. **Add examples** to docstrings where helpful
3. **Update tutorials** for new features
4. **Include mathematical formulations** where appropriate

### Adding New Documentation

1. **API documentation** is generated automatically from docstrings
2. **Tutorials** should be added to `docs/` as Markdown files
3. **Examples** should include both code and expected output

## Submitting Changes

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass:**
   ```bash
   pytest tests/
   pre-commit run --all-files
   ```

4. **Create a pull request** with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to any related issues
   - Screenshots or examples if applicable

### Pull Request Template

When creating a PR, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for this change
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Testing** on different platforms if needed
4. **Documentation review** for user-facing changes

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible  
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Workflow

1. **Update version** in `pyproject.toml` and `__init__.py`
2. **Update CHANGELOG.md** with new version and changes
3. **Create release PR** and get approval
4. **Tag release** on main branch
5. **GitHub Actions** automatically builds and publishes to PyPI

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: [maintainers@chip-project.org](mailto:maintainers@chip-project.org)

### Issue Templates

When reporting bugs or requesting features, please use our issue templates:

- **Bug Report**: Include steps to reproduce, expected vs actual behavior
- **Feature Request**: Describe the feature and its use case
- **Documentation**: Issues with documentation or examples

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub's contributor insights

Thank you for contributing to ASTRA-Torch! 🚀
