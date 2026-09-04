"""Per-dataset acquisition profiles — makes the pipeline reusable across videos.

Everything that differs between time-resolved acquisitions (frame size, rotation
rate, centre-of-rotation, crop, working frame range) lives in a
:class:`DatasetProfile` instead of hardcoded constants. Calibrate a new video
with ``scripts/tr_diffusion_calibrate.py`` (writes a profile JSON), then drive
training / reconstruction with ``--profile <name-or-json>``.

The rotation rate + axis are measured (see ``project-wunderkerze2-rotation``);
only the RATE is calibrated, not the absolute angle of frame 0.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DatasetProfile:
    name: str
    mov_path: str
    fps: float
    height: int
    width: int
    deg_per_frame: float                  # calibrated rotation rate
    rot_axis_col: float                    # calibrated centre-of-rotation (detector column)
    crop: Tuple[int, int]                  # (H, W) fed to denoiser/recon, centred on the axis
    frame_start: int                       # working range within the .mov
    frame_end: int
    memmap_path: Optional[str] = None      # extracted uint16 memmap of the working range
    dose: float = 0.05                     # default extra-noise dose for the study
    norm_range: Optional[Tuple[float, float]] = None  # (norm_min,norm_max); None -> fit from data

    # --- derived rotation quantities ---
    @property
    def period_360(self) -> float:
        return 360.0 / self.deg_per_frame

    @property
    def period_180(self) -> float:
        return 180.0 / self.deg_per_frame

    @property
    def norm_sidecar(self) -> str:
        return str(Path(self.mov_path).with_suffix(".norm.npz"))

    def save(self, path) -> str:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
        return str(path)

    @classmethod
    def load(cls, path_or_name) -> "DatasetProfile":
        """Load from a JSON path, or a registered profile name."""
        p = Path(path_or_name)
        if p.exists():
            d = json.loads(p.read_text())
            d["crop"] = tuple(d["crop"])
            if d.get("norm_range") is not None:
                d["norm_range"] = tuple(d["norm_range"])
            return cls(**d)
        if str(path_or_name) in REGISTRY:
            return REGISTRY[str(path_or_name)]
        raise FileNotFoundError(f"no profile JSON or registered profile named {path_or_name!r}")


_TR = "/myhome/data/sdate/shared/time_resolved"

# Registered profiles (calibrated values). Frame ranges are .mov 0-based indices.
REGISTRY = {
    "wunderkerze2": DatasetProfile(
        name="212_Wunderkerze2",
        mov_path=f"{_TR}/212_Wunderkerze2/212_Wunderkerze2.mov",
        memmap_path=f"{_TR}/212_Wunderkerze2/frames_400k_500k.u16",
        fps=30.0, height=128, width=528,
        deg_per_frame=1.801402, rot_axis_col=269.85,
        crop=(128, 512), frame_start=400_000, frame_end=500_000,
    ),
    "asc_thixo": DatasetProfile(
        name="090_ASC_thixo_650tps",
        mov_path=f"{_TR}/090_ASC_thixo_650tps/090_ASC_thixo_650tps_center_lossless.mov",
        memmap_path=f"{_TR}/090_ASC_thixo_650tps/frames_400k_520k.u16",
        fps=60.0, height=128, width=480,
        deg_per_frame=2.92706, rot_axis_col=235.5,
        crop=(128, 448), frame_start=400_000, frame_end=520_000,
    ),
    "ag10_c1mm": DatasetProfile(
        name="043_AG10_C1mm_0s",
        mov_path=f"{_TR}/043_AG10_C1mm_0s/043_AG10_C1mm_0s_center_lossless.mov",
        memmap_path=f"{_TR}/043_AG10_C1mm_0s/frames_90k_210k.u16",
        fps=60.0, height=280, width=528,
        deg_per_frame=0.900452, rot_axis_col=252.3,
        crop=(256, 480), frame_start=90_000, frame_end=210_000,
    ),
    "synthetic_v1": DatasetProfile(
        name="synthetic_v1",
        mov_path=f"{_TR}/synthetic_v1/synthetic_v1.mov",
        memmap_path=f"{_TR}/synthetic_v1/frames_0_100000.u16",
        fps=30.0, height=128, width=512,
        # Exactly 2 deg/frame -> period_360 = 180 frames EXACTLY (unlike every
        # real dataset above): same-angle temporal taps land on integer frames,
        # no sub-frame interpolation, isolating that confound from the
        # real-dataset ablations. Native width == crop width -> axis_col at the
        # exact centre makes the crop a no-op (see phantom.py's coordinate
        # convention: world 0 <-> pixel (width-1)/2).
        deg_per_frame=2.0, rot_axis_col=511 / 2.0,
        crop=(128, 512), frame_start=0, frame_end=100_000,
        norm_range=(0.0, 700.0),
    ),
}
