"""Setup script for astra-torch library."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="astra-torch",
    version="0.1.0",
    author="CHIP Project",
    author_email="chip@example.com",
    description="GPU-accelerated tomographic reconstruction library with PyTorch integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chip-project/astra-torch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "tqdm>=4.60.0",
        "astra-toolbox>=2.0.0",
        "h5py>=3.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.3.0",
            "ipywidgets>=7.6.0",
        ]
    },
    package_data={
        "astra_torch": ["*.md"],
    },
    include_package_data=True,
)
