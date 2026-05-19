"""Create HEVC gray10 videos from LaDiff reconstruction volumes.

The defaults mirror notebooks/analyze_mrc_volume_reconstructions.ipynb:
matched cases are discovered under the compression-paper data root, volumes
are normalized with the la_fourier norm sidecar, and the circular support mask
is applied before streaming slices as video frames.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdate.stream_hvec.stream_gray10 import EncoderParams, HevcGray10Streamer
from ladiff.fourier_wedge import apply_circle_mask as torch_apply_circle_mask
from ladiff.recon_utils import inpaint_guidance as _inpaint_guidance

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(x, **kwargs):
        return x


DEFAULT_DATA_ROOT = Path("/myhome/data/sdate/shared/compression_paper")
DEFAULT_METHODS = ("diffusion_iterative", "la_fourier", "gt_fbp", "isonet")
MRC_PATTERN = "volume*_rec.mrc"
ANGLE_CORRECTION_DEG = -36.0
ANGULAR_RANGE_FRAC = 0.6
ANGULAR_RANGE_DEG = int(ANGULAR_RANGE_FRAC * 180)
START_ANGLE = (180 - ANGULAR_RANGE_DEG) // 2
TILT_AXIS = 0
INPAINT_TIMESTEP = 0
INPAINT_STEPS = 1

METHOD_ALIASES = {
    "diffusion": "diffusion_iterative",
    "diffusion_iterative": "diffusion_iterative",
    "la": "la_fourier",
    "la_fourier": "la_fourier",
    "fourier": "la_fourier",
    "gt": "gt_fbp",
    "ground_truth": "gt_fbp",
    "gt_fbp": "gt_fbp",
    "isonet": "mrc_volume_rec",
    "isonet_inpainted": "mrc_volume_rec_inpainted",
    "isonet_volume": "mrc_volume_rec",
    "isonet_volume_inpainted": "mrc_volume_rec_inpainted",
    "mrc": "mrc_volume_rec",
    "mrc_inpainted": "mrc_volume_rec_inpainted",
    "mrc_volume_rec": "mrc_volume_rec",
    "mrc_volume_rec_inpainted": "mrc_volume_rec_inpainted",
}


@dataclass(frozen=True)
class VolumeCase:
    file_idx: int
    gt_path: Path
    diffusion_path: Path
    la_fourier_path: Path
    norm_path: Path
    mrc_path: Path | None = None


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def install_ffmpeg() -> None:
    """Install ffmpeg with a common system package manager."""
    if command_exists("apt-get"):
        prefix = [] if os.geteuid() == 0 else ["sudo"]
        commands = [
            prefix + ["apt-get", "update"],
            prefix + ["apt-get", "install", "-y", "ffmpeg"],
        ]
    elif command_exists("dnf"):
        prefix = [] if os.geteuid() == 0 else ["sudo"]
        commands = [prefix + ["dnf", "install", "-y", "ffmpeg"]]
    elif command_exists("yum"):
        prefix = [] if os.geteuid() == 0 else ["sudo"]
        commands = [prefix + ["yum", "install", "-y", "ffmpeg"]]
    elif command_exists("apk"):
        commands = [["apk", "add", "ffmpeg"]]
    elif command_exists("brew"):
        commands = [["brew", "install", "ffmpeg"]]
    else:
        raise RuntimeError(
            "ffmpeg is not installed, and no supported package manager "
            "(apt-get, dnf, yum, apk, brew) was found."
        )

    for cmd in commands:
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def ensure_ffmpeg(auto_install: bool = True) -> None:
    if command_exists("ffmpeg"):
        return
    if not auto_install:
        raise RuntimeError("ffmpeg executable not found")
    print("ffmpeg executable not found; installing it now.", flush=True)
    install_ffmpeg()
    if not command_exists("ffmpeg"):
        raise RuntimeError("ffmpeg installation completed, but ffmpeg is still not on PATH")


def file_idx_from_mrc(path: Path) -> int:
    match = re.search(r"volume(\d+)_rec\.mrc$", path.name)
    if not match:
        raise ValueError(f"Could not parse file_idx from {path}")
    return int(match.group(1))


def mrc_memmap(path: Path) -> tuple[np.memmap, dict[str, int | str]]:
    dtype_by_mode = {
        0: "i1",
        1: "i2",
        2: "f4",
        6: "u2",
        12: "f2",
    }

    with path.open("rb") as f:
        header = f.read(1024)
    if len(header) != 1024:
        raise ValueError(f"{path} is too small to be an MRC file")

    parsed = None
    for endian in ("<", ">"):
        nx, ny, nz, mode = struct.unpack(endian + "4i", header[:16])
        if nx > 0 and ny > 0 and nz > 0 and mode in dtype_by_mode:
            nsymbt = struct.unpack(endian + "i", header[92:96])[0]
            mapc, mapr, maps = struct.unpack(endian + "3i", header[64:76])
            parsed = {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "mode": mode,
                "nsymbt": max(nsymbt, 0),
                "mapc": mapc,
                "mapr": mapr,
                "maps": maps,
                "endian": endian,
            }
            break
    if parsed is None:
        raise ValueError(f"Could not parse MRC header for {path}")

    dtype = np.dtype(dtype_by_mode[int(parsed["mode"])]).newbyteorder(str(parsed["endian"]))
    offset = 1024 + int(parsed["nsymbt"])
    shape = (int(parsed["nz"]), int(parsed["ny"]), int(parsed["nx"]))
    arr = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="C")
    return arr, parsed


def align_volume_to_gt(volume: np.ndarray, gt_shape: tuple[int, int, int]) -> tuple[np.ndarray, str]:
    if tuple(volume.shape) == tuple(gt_shape):
        return volume, "as_read"

    for perm in itertools.permutations(range(3)):
        if tuple(volume.shape[i] for i in perm) == tuple(gt_shape):
            return np.transpose(volume, perm), f"transpose{perm}"

    raise ValueError(f"Cannot align volume shape {volume.shape} to GT shape {gt_shape}")


def rotate_batch(x: np.ndarray, alpha: float, desc: str = "rotate_batch") -> np.ndarray:
    if alpha is None or abs(float(alpha)) < 1e-12:
        return np.asarray(x, dtype=np.float32)

    from scipy.ndimage import rotate

    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for i in tqdm(range(x.shape[0]), desc=desc, leave=False):
        out[i] = rotate(
            x[i],
            angle=float(alpha),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32, copy=False)
    return out


def circle_mask_2d(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def parse_file_idxs(values: Iterable[str] | None) -> set[int] | None:
    if not values:
        return None
    out: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.add(int(part))
    return out


def discover_cases(data_root: Path, file_idxs: set[int] | None = None) -> list[VolumeCase]:
    mrc_by_idx = {file_idx_from_mrc(p): p for p in data_root.rglob(MRC_PATTERN)}
    cases: list[VolumeCase] = []
    for gt_path in sorted(data_root.rglob("gt_fbp_*.npy")):
        match = re.search(r"gt_fbp_(\d+)\.npy$", gt_path.name)
        if not match:
            continue
        file_idx = int(match.group(1))
        if file_idxs is not None and file_idx not in file_idxs:
            continue

        recon_dir = gt_path.parent
        diffusion_path = recon_dir / f"diffusion_{file_idx}_iterative.npy"
        la_fourier_path = recon_dir / f"la_fourier_{file_idx}.npy"
        norm_path = recon_dir / f"la_fourier_{file_idx}_norm.json"
        required = (diffusion_path, la_fourier_path, norm_path)
        if all(p.exists() for p in required):
            cases.append(
                VolumeCase(
                    file_idx=file_idx,
                    gt_path=gt_path,
                    diffusion_path=diffusion_path,
                    la_fourier_path=la_fourier_path,
                    norm_path=norm_path,
                    mrc_path=mrc_by_idx.get(file_idx),
                )
            )
    return cases


def normalize_methods(methods: Iterable[str]) -> list[str]:
    out: list[str] = []
    for method in methods:
        key = method.lower().strip()
        if key not in METHOD_ALIASES:
            choices = ", ".join(sorted(METHOD_ALIASES))
            raise ValueError(f"Unknown method {method!r}. Choices: {choices}")
        canonical = METHOD_ALIASES[key]
        if canonical not in out:
            out.append(canonical)
    return out


def source_for_method(case: VolumeCase, method: str) -> Path:
    if method == "diffusion_iterative":
        return case.diffusion_path
    if method == "la_fourier":
        return case.la_fourier_path
    if method == "gt_fbp":
        return case.gt_path
    if method in {"mrc_volume_rec", "mrc_volume_rec_inpainted"}:
        if case.mrc_path is None:
            raise FileNotFoundError(f"No MRC volume found for file_idx {case.file_idx}")
        return case.mrc_path
    raise ValueError(f"Unhandled method: {method}")


def default_output_path(source_path: Path, method: str) -> Path:
    stem = source_path.stem
    if method in {"mrc_volume_rec", "mrc_volume_rec_inpainted"}:
        stem = f"isonet_{stem}_rot{int(ANGLE_CORRECTION_DEG)}"
        if method == "mrc_volume_rec_inpainted":
            stem = f"{stem}_inpainted"
    return source_path.with_name(f"{stem}_gray10.mov")


def output_path_for(
    source_path: Path,
    method: str,
    output_path: Path | None,
    multiple_outputs: bool,
) -> Path:
    if output_path is None:
        return default_output_path(source_path, method)
    if multiple_outputs or output_path.suffix.lower() not in {".mov", ".mp4", ".mkv"}:
        return output_path / default_output_path(source_path, method).name
    return output_path


def load_norm_range(norm_path: Path) -> tuple[float, float]:
    with norm_path.open() as f:
        norm_cfg = json.load(f)
    norm_min = float(norm_cfg["norm_min"])
    norm_max = float(norm_cfg["norm_max"])
    if not np.isfinite(norm_min) or not np.isfinite(norm_max) or norm_max <= norm_min:
        raise ValueError(f"Invalid normalization range in {norm_path}: {norm_cfg}")
    return norm_min, norm_max


def load_mrc_volume(case: VolumeCase) -> np.ndarray:
    gt = np.load(case.gt_path, mmap_mode="r")
    mrc_path = source_for_method(case, "mrc_volume_rec")
    mrc_raw, _ = mrc_memmap(mrc_path)
    mrc_aligned, alignment = align_volume_to_gt(mrc_raw, tuple(gt.shape))
    print(
        f"file_idx {case.file_idx}: MRC shape {mrc_raw.shape}, "
        f"GT shape {gt.shape}, alignment={alignment}, rotation={ANGLE_CORRECTION_DEG:g} deg",
        flush=True,
    )
    return rotate_batch(
        mrc_aligned,
        ANGLE_CORRECTION_DEG,
        desc=f"{case.file_idx} mrc rotate {ANGLE_CORRECTION_DEG:g} deg",
    )


def inpaint_reconstruction_volume(
    case: VolumeCase,
    recon_np: np.ndarray,
    *,
    apply_circle_mask: bool,
    device: torch.device,
) -> np.ndarray:
    """Apply the notebook's Fourier-wedge inpainting guidance to ISONET."""
    print(
        f"file_idx {case.file_idx}: applying ISONET inpainting guidance on {device} "
        f"(angular_range={ANGULAR_RANGE_DEG}, start={START_ANGLE}, tilt_axis={TILT_AXIS})",
        flush=True,
    )
    gt = np.load(case.gt_path, mmap_mode="r")
    source = torch.as_tensor(np.asarray(gt, dtype=np.float32), device=device)
    recon = torch.as_tensor(np.asarray(recon_np, dtype=np.float32), device=device)

    if apply_circle_mask:
        source = torch_apply_circle_mask(source)
        recon = torch_apply_circle_mask(recon)

    with torch.no_grad():
        guided = _inpaint_guidance(
            source_volume=source,
            vol_init=recon,
            angular_range_deg=ANGULAR_RANGE_DEG,
            start_angle=START_ANGLE,
            tilt_axis=TILT_AXIS,
            device=device,
            timestep=INPAINT_TIMESTEP,
            inpaint_steps=INPAINT_STEPS,
        )
    guided_np = guided.detach().cpu().numpy().astype(np.float32, copy=False)

    del source, recon, guided
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return guided_np


def load_volume(
    case: VolumeCase,
    method: str,
    *,
    apply_circle_mask: bool,
    inpaint_device: torch.device,
) -> np.ndarray:
    if method in {"mrc_volume_rec", "mrc_volume_rec_inpainted"}:
        mrc_rotated = load_mrc_volume(case)
        if method == "mrc_volume_rec_inpainted":
            return inpaint_reconstruction_volume(
                case,
                mrc_rotated,
                apply_circle_mask=apply_circle_mask,
                device=inpaint_device,
            )
        return mrc_rotated
    return np.load(source_for_method(case, method), mmap_mode="r")


def frame_to_tensor(
    frame: np.ndarray,
    norm_min: float,
    norm_max: float,
    mask: np.ndarray | None,
) -> torch.Tensor:
    arr = np.asarray(frame, dtype=np.float32)
    arr = (arr - norm_min) / (norm_max - norm_min)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    if mask is not None:
        arr = np.where(mask, arr, 0.0)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def encode_volume(
    volume: np.ndarray,
    output_path: Path,
    norm_min: float,
    norm_max: float,
    *,
    apply_circle_mask: bool,
    fps: int,
    crf: int,
    preset: str,
    threads: int,
    force_software: bool,
) -> None:
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape (D, H, W), got {volume.shape}")

    mask = circle_mask_2d(tuple(volume.shape[-2:])) if apply_circle_mask else None
    params = EncoderParams(
        fps=fps,
        crf_sw=crf,
        preset_sw=preset,
        force_software=force_software,
        threads=threads,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    streamer = HevcGray10Streamer(
        base_path=output_path.parent,
        segment_prefix=output_path.stem,
        params=params,
    )
    with streamer.start_segment(outfile=output_path.name):
        for z in tqdm(range(volume.shape[0]), desc=f"encode {output_path.name}", leave=False):
            streamer.append_frame(frame_to_tensor(volume[z], norm_min, norm_max, mask))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create HEVC gray10 videos from diffusion, LA Fourier, GT, and optional MRC volumes."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--file-idxs", nargs="*", help="Optional file IDs, e.g. --file-idxs 1 10")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Methods to encode: diffusion_iterative, la_fourier, gt_fbp, isonet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output directory, or a single video path when encoding exactly one output.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--no-circle-mask",
        action="store_true",
        help="Disable the circular support mask used by the notebook evaluation path.",
    )
    parser.add_argument(
        "--inpaint-isonet",
        action="store_true",
        help="Also save an ISONET video after notebook-style Fourier-wedge inpainting guidance.",
    )
    parser.add_argument(
        "--inpaint-device",
        default=None,
        help="Device for ISONET inpainting guidance, e.g. cuda or cpu. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--allow-hardware",
        action="store_true",
        help="Allow the streamer to try its hardware path before libx265 software encoding.",
    )
    parser.add_argument(
        "--no-install-ffmpeg",
        action="store_true",
        help="Only check for ffmpeg and fail if it is missing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ensure_ffmpeg(auto_install=not args.no_install_ffmpeg)

    file_idxs = parse_file_idxs(args.file_idxs)
    methods = normalize_methods(args.methods)
    if args.inpaint_isonet and "mrc_volume_rec" in methods and "mrc_volume_rec_inpainted" not in methods:
        methods.append("mrc_volume_rec_inpainted")

    apply_circle_mask = not args.no_circle_mask
    inpaint_device = torch.device(
        args.inpaint_device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cases = discover_cases(args.data_root, file_idxs=file_idxs)
    if not cases:
        raise RuntimeError(f"No complete cases found under {args.data_root}")

    planned_outputs = [
        (case, method, source_for_method(case, method))
        for case in cases
        for method in methods
    ]
    multiple_outputs = len(planned_outputs) > 1

    print(
        f"Found {len(cases)} case(s); encoding {len(planned_outputs)} video(s).",
        flush=True,
    )
    if "mrc_volume_rec_inpainted" in methods:
        print(f"ISONET inpainting device: {inpaint_device}", flush=True)
    for case, method, source_path in planned_outputs:
        norm_min, norm_max = load_norm_range(case.norm_path)
        output_path = output_path_for(source_path, method, args.output_path, multiple_outputs)
        volume = load_volume(
            case,
            method,
            apply_circle_mask=apply_circle_mask,
            inpaint_device=inpaint_device,
        )
        print(
            f"file_idx {case.file_idx}: {method} {tuple(volume.shape)} "
            f"{volume.dtype} -> {output_path}",
            flush=True,
        )
        encode_volume(
            volume,
            output_path,
            norm_min,
            norm_max,
            apply_circle_mask=apply_circle_mask,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            threads=args.threads,
            force_software=not args.allow_hardware,
        )
        print(f"wrote {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
