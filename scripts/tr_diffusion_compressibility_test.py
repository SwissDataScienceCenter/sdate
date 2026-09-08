#!/usr/bin/env python3
"""Quick test: how much more HEVC-compressible is the denoised jointfbpctx_T11
projection stream than the raw noisy (dose=0.05) measurement it was built
from?

Both streams are encoded FRESH from first-generation raw data (not re-encoded
from an already-compressed movie, which would double-compress and bias the
comparison): the denoised stream is read directly from its stored float16
memmap; the noisy stream is regenerated on the fly from the native frames
using the SAME dose/seed convention used everywhere in this project
(noisy_window_gpu, dose=0.05, seed=12345) -- statistically identical to what
the model was actually shown, though not necessarily bit-identical to any
previously-rendered movie's noisy panel (different chunk offset).

    python scripts/tr_diffusion_compressibility_test.py
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

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

DENOISED_MM = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/denoised_212_Wunderkerze2_jointfbpctx_T11_full_dose05.f16"
OUT_DIR = Path("/myhome/data/sdate/shared/time_resolved/tr_recon_cache/compressibility_test")
FRAME_START = 420_000
N = 500
DOSE, NOISE_SEED = 0.05, 12345
PRESET = "medium"  # veryslow timed out on this shared CPU sandbox; medium is a reasonable speed/quality tradeoff for a RELATIVE comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prof = DatasetProfile.load("wunderkerze2")
    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)

    idx = np.arange(FRAME_START, FRAME_START + N)
    log(f"loading native frames {FRAME_START}..{FRAME_START + N}")
    native = R.native_window_gpu(src, idx, prof.crop, prof.rot_axis_col, device)

    log("synthesizing matched noisy stream (dose=0.05, seed=12345)")
    gen = torch.Generator(device=device).manual_seed(NOISE_SEED + FRAME_START)
    noisy = R.noisy_window_gpu(native, DOSE, generator=gen)

    log("reading denoised stream directly from stored memmap")
    meta = np.load(DENOISED_MM + ".meta.npz")
    first_index, num_frames = int(meta["first_index"]), int(meta["num_frames"])
    crop = (int(meta["crop"][0]), int(meta["crop"][1]))
    mm = np.memmap(DENOISED_MM, dtype=np.float16, mode="r", shape=(num_frames, *crop))
    denoised_np = np.asarray(mm[idx - first_index]).astype(np.float32)
    denoised = torch.from_numpy(denoised_np).to(device)

    # shared normalization (percentile of the noisy stream, which has the widest
    # range) so both encodes quantize on the same intensity scale
    vmin, vmax = np.percentile(noisy.cpu().numpy(), [1, 99])
    span = max(vmax - vmin, 1e-6)
    log(f"shared normalization: vmin={vmin:.2f} vmax={vmax:.2f}")

    params = EncoderParams(force_software=True, crf_sw=14, preset_sw=PRESET)

    for tag, stack in (("noisy", noisy), ("denoised", denoised)):
        out_path = OUT_DIR / f"test_{tag}.mov"
        t0 = time.time()
        st = HevcGray10Streamer(OUT_DIR, segment_prefix=f"test_{tag}_seg", params=params)
        with st.start_segment(q=90):
            for i in range(stack.shape[0]):
                frame = ((stack[i].float() - vmin) / span).clamp(0.0, 1.0)
                st.append_frame(frame.detach().cpu().contiguous())
        from sdate.stream_hvec.stream_gray10 import concat_hevc_segments
        concat_hevc_segments(st.segments, str(out_path))
        size = out_path.stat().st_size
        mins = (time.time() - t0) / 60
        log(f"{tag:10s} {N} frames -> {size / 1e6:.1f} MB  ({size / N:.0f} bytes/frame)  [{mins:.1f} min]")

    raw_bytes_per_frame = crop[0] * crop[1] * 2  # 10-bit-in-16-bit raw
    noisy_size = (OUT_DIR / "test_noisy.mov").stat().st_size
    denoised_size = (OUT_DIR / "test_denoised.mov").stat().st_size
    log("=== SUMMARY ===")
    log(f"raw uncompressed:      {raw_bytes_per_frame} bytes/frame")
    log(f"noisy compressed:      {noisy_size / N:.0f} bytes/frame  ({raw_bytes_per_frame * N / noisy_size:.1f}x raw)")
    log(f"denoised compressed:   {denoised_size / N:.0f} bytes/frame  ({raw_bytes_per_frame * N / denoised_size:.1f}x raw)")
    log(f"denoised is {noisy_size / denoised_size:.2f}x MORE compressible than noisy (same codec/settings)")
    log("DONE")


if __name__ == "__main__":
    main()
