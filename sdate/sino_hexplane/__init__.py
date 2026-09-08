"""Sinogram-domain HexPlane INR for tomoscopy projection denoising.

Fits an implicit neural representation directly in projection/sinogram space
(no forward model, no tomographic reconstruction): a HexPlane / K-Planes
factorisation of the 4-D coordinate ``(theta, s, v, t)`` into six pairwise 2-D
feature planes, Hadamard-combined and decoded to a line integral ``p_hat``,
trained with a count-space Poisson loss.  See the ``project-sino-hexplane``
memory / project spec for design and decisions.
"""

from .planes import Plane2D
from .model import HexPlaneEncoding, ProjectionField, default_plane_config
from .encodings import JointHashEncoding
from .data import HelixFrame, HelixSinoDataset, build_helix_noise_acquisition
from .train import build_field_for, train_projection_field, predict, predict_frame, predict_sino
from .metrics import frame_metrics, evaluate_all_frames, psnr
from .reconstruct import (gd_reconstruct, reconstruct_frame, reconstruct_and_evaluate,
                          ARMS, DEFAULT_GD)
from .movies import generate_projection_movie

__all__ = [
    "Plane2D", "HexPlaneEncoding", "JointHashEncoding", "ProjectionField", "default_plane_config",
    "HelixFrame", "HelixSinoDataset", "build_helix_noise_acquisition",
    "build_field_for", "train_projection_field", "predict", "predict_frame", "predict_sino",
    "frame_metrics", "evaluate_all_frames", "psnr",
    "gd_reconstruct", "reconstruct_frame", "reconstruct_and_evaluate", "ARMS", "DEFAULT_GD",
    "generate_projection_movie",
]
