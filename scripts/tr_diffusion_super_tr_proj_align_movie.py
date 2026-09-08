#!/usr/bin/env python3
"""Time-aligned projection-stream + reconstruction movie, for visually checking
whether motion appearing in a super-TR reconstruction sweep (see
tr_diffusion_super_tr_movie.py) is actually visible in the underlying raw
projections, or an artifact of the synthesis.

Each reconstruction "chunk" (one output position from an existing
super_tr_movie sweep, identified by its `_slices.pt`) corresponds to a
20-frame (stride) real window [c, c+stride). --n_proj_samples (default 1)
controls how many real frames from that window are shown per chunk:
- 1 (default): just the window's CENTRAL frame (c + stride//2). Since c
  itself advances by only 1 real frame per output step (chunks overlap by
  stride-1 out of stride frames), this makes the projection stream advance
  monotonically forward in lock-step with the reconstruction stream -- no
  within-chunk motion to show, so no jumps.
- >1: --n_proj_samples evenly-spaced frames from within the window, played
  while the reconstruction panel is held fixed (n_chunks * n_proj_samples
  total video frames). NOTE: because consecutive chunks overlap so heavily,
  this reads as a sawtooth -- forward within each chunk, then a ~stride-frame
  rewind at every chunk boundary (confirmed with the user, 2026-09-05) --
  it shows the full within-chunk real span, but is confusing to watch as
  straight playback.

Layout per video frame: top strip = the sampled raw projection frame (native
counts, percentile-normalized once from the first sample and held fixed,
matching the convention in reconstruct.write_projection_movie), tiled twice
to fill the width; bottom row = predicted | reference reconstruction slices
(same normalization as the original super_tr_movie sweep, from the `ref`
stack's own percentile).

Runs entirely on raw counts + simple tensor ops + video encoding -- no FBP,
no model inference -- light enough for a CPU-only machine, no GPU needed.

  python scripts/tr_diffusion_super_tr_proj_align_movie.py \\
      --base_tag 212_Wunderkerze2_super_tr_c436000_n600
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.frames import MemmapFrameSource  # noqa: E402
from sdate.tr_diffusion.profiles import DatasetProfile  # noqa: E402
from sdate.stream_hvec.stream_gray10 import EncoderParams, HevcGray10Streamer, concat_hevc_segments  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--base_tag", required=True,
                   help="tag of an existing tr_diffusion_super_tr_movie.py sweep -- reads "
                        "{recon_dir}/{base_tag}_slices.pt for predicted/reference slices + c_positions")
    p.add_argument("--recon_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--n_proj_samples", type=int, default=1,
                   help="1 = just the window's central frame, advancing monotonically in lock-step "
                        "with the reconstruction stream (recommended); >1 = evenly-spaced frames "
                        "spanning the window, which saw-tooths at chunk boundaries since consecutive "
                        "chunks overlap heavily (see module docstring)")
    p.add_argument("--out", default=None)
    p.add_argument("--q", type=int, default=90)
    p.add_argument("--preset_sw", default="medium")
    p.add_argument("--log_every", type=int, default=50,
                   help="log progress every N reconstruction (chunk) positions")
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recon_dir = Path(a.recon_dir)
    t0 = time.time()

    slices_path = recon_dir / f"{a.base_tag}_slices.pt"
    data = torch.load(slices_path, map_location="cpu", weights_only=False)
    pred, ref, c_positions = data["pred"], data["ref"], np.asarray(data["c_positions"])
    n = len(c_positions)
    print(f"loaded {slices_path}: {n} chunks, c in [{c_positions[0]},{c_positions[-1]}], "
          f"pred shape {tuple(pred.shape)}", flush=True)

    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)

    out_path = Path(a.out) if a.out else recon_dir / f"{a.base_tag}_proj_aligned.mov"
    st = HevcGray10Streamer(out_path.parent, segment_prefix=out_path.stem,
                            params=EncoderParams(force_software=True, preset_sw=a.preset_sw))

    # recon-panel normalization: same convention as the original sweep (percentile of
    # the whole `ref` stack), fixed across the whole movie.
    vmin_r = float(np.percentile(ref.numpy(), 1))
    vmax_r = float(np.percentile(ref.numpy(), 99))
    span_r = max(vmax_r - vmin_r, 1e-6)

    if a.n_proj_samples == 1:
        offsets = np.array([a.stride // 2])
    else:
        offsets = np.linspace(0, a.stride - 1, a.n_proj_samples).round().astype(int)
    print(f"sampling {a.n_proj_samples} projection frames per chunk at offsets {offsets.tolist()} "
          f"within each [c,c+{a.stride}) window", flush=True)

    proj_vmin = proj_vmax = proj_span = None
    with st.start_segment(q=a.q):
        for i in range(n):
            c = int(c_positions[i])
            # native_window_gpu does ONE contiguous memmap read spanning [idx[0],idx[-1]] --
            # passing the 5 non-contiguous sample offsets directly would silently read back
            # the full 20-frame window anyway (see feedback_native_window_gpu_contiguous_only
            # in project memory), so fetch the whole window once and slice the samples out.
            full_idx = np.arange(c, c + a.stride)
            full_native = R.native_window_gpu(src, full_idx, prof.crop, prof.rot_axis_col, device).float()
            proj_native = full_native[offsets]
            if proj_vmin is None:
                lo_p, hi_p = np.percentile(proj_native.cpu().numpy(), [1, 99])
                proj_vmin, proj_vmax = float(lo_p), float(hi_p)
                proj_span = max(proj_vmax - proj_vmin, 1e-6)
            proj_norm = ((proj_native - proj_vmin) / proj_span).clamp(0.0, 1.0).cpu()

            pred_n = ((pred[i].float() - vmin_r) / span_r).clamp(0.0, 1.0)
            ref_n = ((ref[i].float() - vmin_r) / span_r).clamp(0.0, 1.0)
            recon_row = torch.cat([pred_n, ref_n], dim=1)  # (H_r, 2*W_r)

            for j in range(a.n_proj_samples):
                proj_strip = torch.cat([proj_norm[j], proj_norm[j]], dim=1)  # (H_p, 2*W_r)
                frame = torch.cat([proj_strip, recon_row], dim=0)  # (H_p+H_r, 2*W_r)
                st.append_frame(frame.contiguous())

            if a.log_every and (i + 1) % a.log_every == 0:
                print(f"  {i + 1}/{n} chunks -> {(i + 1) * a.n_proj_samples} video frames  "
                      f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    concat_hevc_segments(st.segments, str(out_path))
    total_frames = n * a.n_proj_samples
    print(f"movie written -> {out_path}  {total_frames} total frames "
          f"({a.n_proj_samples} projection sub-frames per reconstruction chunk, "
          f"top strip=raw projection (tiled) | bottom row=predicted|reference)", flush=True)
    print(f"TOTAL runtime: {(time.time() - t0) / 60:.1f} min", flush=True)
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
