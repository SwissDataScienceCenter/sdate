# Examples

This directory contains practical examples demonstrating ASTRA-Torch functionality.

## Python Scripts

### basic_cbct_example.py
Demonstrates basic cone-beam CT reconstruction using the FDK algorithm:
- Creating circular acquisition geometry
- Simulating projection data
- FDK reconstruction
- Visualization of results

Run with:
```bash
python basic_cbct_example.py
```

### laminography_example.py  
Shows laminography reconstruction with filtered backprojection:
- Creating tilted-axis geometry
- Layered phantom construction
- FBP reconstruction
- Multi-slice visualization

Run with:
```bash
python laminography_example.py
```

### parallel2d_example.py
Demonstrates 2D parallel beam tomographic reconstruction:
- Creating Shepp-Logan phantom
- Generating parallel beam projections
- Comparing FBP, SIRT, and gradient descent methods
- Quality metrics and visualization

Run with:
```bash
python parallel2d_example.py
```

## Jupyter Notebooks

### cbct_tutorial.ipynb
Interactive tutorial covering:
- CBCT geometry fundamentals
- Forward and backward projection
- Comparison of FDK vs gradient descent
- Parameter optimization

### advanced_reconstruction.ipynb
Advanced techniques including:
- Custom loss functions
- Regularization methods
- Multi-GPU reconstruction
- Memory optimization

### walnut_dataset_demo.ipynb
Real data example using the Walnut dataset:
- Data loading and preprocessing
- Quality assessment
- Artifact reduction
- Results comparison

## Running the Examples

### Prerequisites
```bash
pip install astra-torch[notebooks]  # For notebook examples
conda install -c astra-toolbox astra-toolbox  # ASTRA dependency
```

### Python Scripts
All Python scripts can be run directly:
```bash
cd examples
python basic_cbct_example.py
python laminography_example.py
```

### Jupyter Notebooks
Start Jupyter and open the notebooks:
```bash
cd examples
jupyter notebook
```

## Expected Output

Each example saves visualization results:
- `basic_cbct_example.py` → `cbct_reconstruction_example.png`
- `laminography_example.py` → `laminography_example.png`
- Notebooks → Interactive plots and saved figures

## Troubleshooting

**ASTRA Import Errors**: Install ASTRA toolbox via conda
**CUDA Issues**: Examples work on CPU but benefit from GPU acceleration
**Memory Errors**: Reduce volume/detector resolution in examples

For more help, see the [Installation Guide](../docs/installation.md).
