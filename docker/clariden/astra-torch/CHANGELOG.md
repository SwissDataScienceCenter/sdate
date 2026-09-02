# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with pytest configuration
- CI/CD workflows for testing, linting, and deployment  
- Code quality tools (flake8, mypy, black, pre-commit)
- Sphinx documentation with ReadTheDocs integration
- Example scripts and notebooks
- Security policy and dependency management
- Contribution guidelines

### Changed
- Enhanced project structure for deployment readiness
- Improved package configuration in pyproject.toml

### Fixed
- Package metadata and dependencies

## [0.1.0] - 2024-01-XX

### Added
- Initial release of ASTRA-Torch
- CBCT reconstruction with FDK and gradient descent algorithms
- Laminography reconstruction with FBP and gradient descent algorithms
- PyTorch integration with automatic differentiation
- GPU acceleration via ASTRA toolbox
- Support for Walnut dataset format
- Basic documentation and examples

### Features
- `CBCTAcquisition` class for cone-beam CT geometry management
- `LaminoAcquisition` class for laminography geometry management
- Forward projection operators for both CBCT and laminography
- Masked reconstruction functions for memory efficiency
- Flexible parameter configuration for reconstruction algorithms
