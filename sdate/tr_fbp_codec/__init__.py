"""tr_fbp_codec -- learned lossless codec for time-resolved CT projection streams.

Predicts frame f[n+k] from context frames f[n..n+k-1] plus an FBP-reprojected
prior tap, trained with cross-entropy over 12-bit quantized pixel values,
entropy-coded via arithmetic coding on the predicted distribution.

See README.md for the design summary and DEPENDENCIES.md for the exact
coupling points with other sdate packages (tr_diffusion in particular).
This package is otherwise self-contained: do not import from it in other
sdate packages without adding the reverse dependency to DEPENDENCIES.md.
"""

__version__ = "0.1.0"
