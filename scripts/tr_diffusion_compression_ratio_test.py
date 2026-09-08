#!/usr/bin/env python3
"""Lossless HEVC (gray10) compression-ratio test: native (raw, noisy) projections
vs. the jointfbpctx_T5_native_gaussianfloor-denoised projections, over the same
~1000-frame window, encoded independently with sdate.stream_hvec.stream_gray10.

Motivation: the fine-tuned T5-native denoiser and the raw native floor are
reconstructed-domain "almost identical, qualitatively hard to differentiate"
-- if the denoised PROJECTION stream is measurably smoother (less independent
per-pixel noise), it should compress noticeably better under a fixed lossless
codec even though it looks the same to the eye. This script tests that.

Window: 1000 frames centred on frame 439900 -- the middle of the eval movie's
frame range [414900,464900] and the documented PSNR-vs-time "most dynamic"
trough (see project memory project-tr-diffusion-t5native-gaussianfloor.md).

Precision note: stream_gray10's software path downconverts to gray10le (10-bit,
1024 levels) regardless of the 16-bit input pipe -- ffmpeg's format filter does
a plain bit-depth downconversion (roughly value>>6), NOT a percentile rescale.
Raw detector counts here (~tens-to-hundreds) would collapse to near-zero if fed
through the library's naive /65535 convention, losing virtually all structure.
So both streams are affinely rescaled by the SAME shared (vmin, vmax) -- taken
from the native raw window (which has >= the denoised window's dynamic range,
since it carries extra noise variance around the same signal) -- to use the
full available 10-bit precision. "Lossless" here means lossless AT that shared
10-bit quantisation, which is the ceiling this codec path offers; quantisation
is applied identically to both arms so the comparison stays fair.

    python scripts/tr_diffusion_compression_ratio_test.py
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.frames import MemmapFrameSource  # noqa: E402
from sdate.tr_diffusion.profiles import DatasetProfile  # noqa: E402
from sdate.stream_hvec.stream_gray10 import EncoderParams, HevcGray10Streamer  # noqa: E402

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
OUT_DIR = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache/compression_test")
DENOISED_MM = f"{DATA}/denoised_212_Wunderkerze2_jointfbpctx_T5_native_gaussianfloor.f16"

CENTER = 439_900  # middle of the eval movie's frame range [414900,464900] -- also
                   # the documented PSNR-vs-time "most dynamic" trough
N_FRAMES = 1000
LO, HI = CENTER - N_FRAMES // 2, CENTER - N_FRAMES // 2 + N_FRAMES

SUMMARY_PATH = OUT_DIR / "compression_summary.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def encode_lossless(name: str, arr: np.ndarray, vmin: float, vmax: float) -> dict:
    """arr: (N,H,W) float32 counts. Rescale to [0,1] via shared (vmin,vmax), encode."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile_name = f"{name}_lossless.mov"
    params = EncoderParams(force_software=True, cq_hw=101, preset_sw="veryslow",
                           fps=24, keyint=None)
    streamer = HevcGray10Streamer(base_path=OUT_DIR, segment_prefix=name, params=params)
    t0 = time.time()
    with streamer.start_segment(q=101, outfile=outfile_name):
        for i in range(arr.shape[0]):
            frame01 = np.clip((arr[i] - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
            streamer.append_frame(torch.from_numpy(frame01))
            if i % 200 == 0:
                log(f"  [{name}] frame {i}/{arr.shape[0]}")
    outfile = streamer.segments[-1]
    size_bytes = outfile.stat().st_size
    minutes = (time.time() - t0) / 60
    raw_bytes = arr.astype(np.uint16).nbytes  # uncompressed 16-bit reference size
    log(f"[{name}] done in {minutes:.1f} min -> {outfile}  "
        f"size={size_bytes / 1e6:.2f} MB  raw16bit={raw_bytes / 1e6:.2f} MB  "
        f"ratio={raw_bytes / size_bytes:.2f}x")
    return {
        "file": str(outfile), "size_bytes": size_bytes, "size_mb": size_bytes / 1e6,
        "raw16bit_mb": raw_bytes / 1e6, "compression_ratio_vs_raw16": raw_bytes / size_bytes,
        "minutes": round(minutes, 2), "n_frames": int(arr.shape[0]),
    }


def main() -> None:
    log(f"window: frames [{LO},{HI}) ({N_FRAMES} frames), centre={CENTER}")
    prof = DatasetProfile.load("wunderkerze2")
    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)
    device = torch.device("cpu")

    idx = np.arange(LO, HI)
    raw = R.native_window_gpu(src, idx, prof.crop, prof.rot_axis_col, device).float().numpy()
    log(f"native raw: shape={raw.shape} min={raw.min():.2f} max={raw.max():.2f} mean={raw.mean():.2f}")

    m = np.load(str(DENOISED_MM) + ".meta.npz")
    d_first, d_n = int(m["first_index"]), int(m["num_frames"])
    crop = (int(m["crop"][0]), int(m["crop"][1]))
    assert d_first <= LO and HI <= d_first + d_n, (
        f"denoised memmap [{d_first},{d_first + d_n}) does not cover [{LO},{HI})")
    d_mm = np.memmap(DENOISED_MM, dtype=np.float16, mode="r", shape=(d_n, *crop))
    den = np.asarray(d_mm[idx - d_first]).astype(np.float32)
    log(f"denoised:   shape={den.shape} min={den.min():.2f} max={den.max():.2f} mean={den.mean():.2f}")

    # Shared scale (same affine map for both arms -- fair comparison, see module docstring).
    vmin = float(min(raw.min(), den.min()))
    vmax = float(max(raw.max(), den.max()))
    log(f"shared scale for 10-bit mapping: vmin={vmin:.2f} vmax={vmax:.2f}")

    results = {}
    results["native_raw"] = encode_lossless("native_raw", raw, vmin, vmax)
    results["jointfbpctx_T5_native_gaussianfloor"] = encode_lossless(
        "jointfbpctx_T5_native_gaussianfloor", den, vmin, vmax)

    summary = {
        "center": CENTER, "frame_range": [int(LO), int(HI)], "n_frames": N_FRAMES,
        "shared_scale": {"vmin": vmin, "vmax": vmax},
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    log("SUMMARY " + json.dumps(summary, indent=2))
    log(f"summary written -> {SUMMARY_PATH}")

    ratio_gain = (results["jointfbpctx_T5_native_gaussianfloor"]["size_bytes"]
                  / results["native_raw"]["size_bytes"])
    log(f"denoised/native size ratio: {ratio_gain:.3f}  "
        f"({'denoised SMALLER (compresses better)' if ratio_gain < 1 else 'denoised LARGER'})")
    log("DONE")


if __name__ == "__main__":
    main()
