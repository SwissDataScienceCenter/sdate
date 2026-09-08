#!/usr/bin/env python3
"""Quick raw-projection motion movie for the Sewellia lineolata real dataset.

Purpose: a fast qualitative look at the fish's motion and its amplitude over
the first N frames -- NOT a calibrated/denoised product, just raw counts
streamed straight to video. Mirrors the streaming convention used by
sdate.tr_diffusion.reconstruct.write_projection_movie (chunked read, fixed
percentile normalization from the first chunk, HevcGray10Streamer with
force_software=True) -- see that function's docstring for why streaming
matters: buffering the whole movie in RAM caused a real OOM on the shared
server before.

Frames are read directly from the HDF5 file chunk-by-chunk (h5py slicing
only pulls the requested chunk off disk -- the full (20000,580,576) array is
never materialized), so memory stays at O(chunk), not O(n_frames).

    python scripts/sewellia_projection_motion_movie.py --n_frames 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")

from sdate.stream_hvec.stream_gray10 import EncoderParams, HevcGray10Streamer, concat_hevc_segments

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--phase_txt", default=PHASE_TXT, help="pass '' / omit-equivalent for datasets with no phase sidecar (e.g. plain radiography, no periodic motion tag)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n_frames", type=int, default=2000)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--chunk", type=int, default=200, help="frames read/encoded per chunk -- keeps memory O(chunk)")
    p.add_argument("--q", type=int, default=90)
    p.add_argument("--preset_sw", default="medium", help="x265 preset -- 'veryslow' is unnecessary for a quick exploratory look")
    p.add_argument("--tag", default="raw", help="included in the output filename, so a raw vs. denoised render of the same range don't collide")
    p.add_argument("--vmin", type=float, default=None, help="fix normalization to a specific value (e.g. to match another render exactly) instead of computing from the first chunk")
    p.add_argument("--vmax", type=float, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    lo, hi = a.start, a.start + a.n_frames
    out_path = Path(a.out_dir) / f"sewellia_projections_{a.tag}_motion_f{lo}_{hi}.mov"

    with h5py.File(a.data_path, "r") as f:
        n_proj, h, w = f["exchange/data"].shape
        hi = min(hi, n_proj)
        log(f"dataset n_proj={n_proj} h={h} w={w}; rendering frames [{lo},{hi})")

        if a.phase_txt and Path(a.phase_txt).exists():
            phase = np.loadtxt(a.phase_txt).astype(np.float64)
            seg = phase[lo:hi]
            log(f"phase over this range: min={seg.min():.4f} max={seg.max():.4f} "
                f"span={seg.max()-seg.min():.4f} rad ({np.degrees(seg.max()-seg.min()):.1f} deg) "
                f"-- amplitude proxy for the motion shown in the movie")
        else:
            log("no phase sidecar file -- skipping phase/amplitude readout")

        st = HevcGray10Streamer(out_path.parent, segment_prefix=out_path.stem,
                                params=EncoderParams(fps=a.fps, force_software=True, preset_sw=a.preset_sw))
        vmin = a.vmin
        vmax = a.vmax
        span = max(vmax - vmin, 1e-6) if (vmin is not None and vmax is not None) else None
        if vmin is not None:
            log(f"normalization fixed by caller: vmin={vmin:.1f} vmax={vmax:.1f}")
        t0 = time.time()
        with st.start_segment(q=a.q):
            for c0 in range(lo, hi, a.chunk):
                c1 = min(c0 + a.chunk, hi)
                chunk = f["exchange/data"][c0:c1].astype(np.float32)  # only this chunk touches disk
                if vmin is None:
                    lo_p, hi_p = np.percentile(chunk, [1, 99])
                    vmin, vmax = float(lo_p), float(hi_p)
                    span = max(vmax - vmin, 1e-6)
                    log(f"normalization fixed from first chunk: vmin={vmin:.1f} vmax={vmax:.1f}")
                normed = np.clip((chunk - vmin) / span, 0.0, 1.0)
                for i in range(normed.shape[0]):
                    st.append_frame(torch.from_numpy(normed[i]).contiguous())
                log(f"  encoded [{c0},{c1}) ({c1-lo}/{hi-lo} frames, {time.time()-t0:.1f}s)")
        concat_hevc_segments(st.segments, str(out_path))

    log(f"saved -> {out_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
