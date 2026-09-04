#!/usr/bin/env python3
"""Calibrate a new time-resolved .mov -> a reusable DatasetProfile JSON.

Measures the rotation rate (sub-frame, via local identity-period autocorrelation)
and the centre-of-rotation column (via 180deg mirror NCC), denormalising counts
through the ``.norm.npz`` sidecar, and writes a profile the rest of the pipeline
consumes with ``--profile``. Frame size / fps come from ffprobe if not given.

  python scripts/tr_diffusion_calibrate.py \
    --mov /.../090_ASC_thixo_650tps/090_ASC_thixo_650tps_center_lossless.mov \
    --name 090_ASC_thixo_650tps --frame_start 400000 --frame_end 520000
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/myhome/sdate")
from sdate.tr_diffusion.profiles import DatasetProfile

FFMPEG = "/myhome/bin/ffmpeg" if Path("/myhome/bin/ffmpeg").exists() else "ffmpeg"
FFPROBE = "/myhome/bin/ffprobe" if Path("/myhome/bin/ffprobe").exists() else "ffprobe"


def probe(mov):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", mov],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def largest_mult(n, m=32):
    return int(n // m) * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mov", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--frame_start", type=int, default=400_000)
    ap.add_argument("--frame_end", type=int, default=520_000)
    ap.add_argument("--period_lo", type=int, default=40, help="min plausible frames/turn for the search")
    ap.add_argument("--period_hi", type=int, default=600)
    ap.add_argument("--out", default=None, help="profile JSON path (default: <mov dir>/tr_diffusion_profile.json)")
    a = ap.parse_args()

    W, H, fps = probe(a.mov)
    W = a.width or W; H = a.height or H; fps = a.fps or fps
    FB = 2 * W * H
    side = np.load(str(Path(a.mov).with_suffix(".norm.npz")))
    pmin, pmax = side["per_frame_min"].astype(np.float64), side["per_frame_max"].astype(np.float64)
    print(f"{W}x{H} @ {fps}fps, sidecar frames={len(pmin)}")

    def counts(start, n):
        t = (start + 0.5) / fps
        p = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.6f}", "-i", a.mov, "-frames:v", str(n),
                            "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"], capture_output=True).stdout
        raw = np.frombuffer(p[:FB * n], np.uint16).reshape(n, H, W).astype(np.float64)
        out = np.empty_like(raw)
        for j in range(n):
            out[j] = raw[j] / 65535.0 * (pmax[start + j] - pmin[start + j]) + pmin[start + j]
        return out

    # coarse integer period from one window, then sub-frame parabolic at several points
    prof0 = counts(a.frame_start + (a.frame_end - a.frame_start) // 2, 900).sum(1)
    def ncc(x, y):
        x = x - x.mean(); y = y - y.mean(); return (x @ y) / (np.sqrt((x @ x) * (y @ y)) + 1e-12)
    lags = np.arange(a.period_lo, a.period_hi)
    coarse = lags[np.argmax([np.mean([ncc(prof0[k], prof0[k + P]) for k in range(0, len(prof0) - P, 5)]) for P in lags])]
    fine_lags = np.arange(max(3, coarse - 8), coarse + 9)

    def local_period(start, M=900):
        prof = counts(start, M).sum(1); Pm = prof - prof.mean(1, keepdims=True); nrm = np.sqrt((Pm * Pm).sum(1))
        sc = np.array([((Pm[:-L] * Pm[L:]).sum(1) / (nrm[:-L] * nrm[L:] + 1e-12)).mean() for L in fine_lags])
        j = sc.argmax(); y0, y1, y2 = sc[j - 1], sc[j], sc[j + 1]
        return fine_lags[j] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)

    pts = np.linspace(a.frame_start, a.frame_end - 1000, 12).astype(int)
    per = np.array([local_period(s) for s in pts])
    period = float(per.mean()); dpf = 360.0 / period
    print(f"period {period:.3f} frames/turn (std {per.std():.3f}) -> {dpf:.5f} deg/frame ; {fps/period:.4f} turns/s")

    # centre of rotation via 180deg mirror NCC
    half = int(round(period / 2)); F = counts(a.frame_start + (a.frame_end - a.frame_start) // 2, half + 30).sum(1)
    best = (-1, None)
    for axis in np.arange(W * 0.35, W * 0.65, 0.5):
        src = np.clip(np.round(2 * axis - np.arange(W)).astype(int), 0, W - 1)
        s = np.mean([ncc(F[k], F[k + half][src]) for k in range(0, 20)])
        if s > best[0]: best = (s, float(axis))
    axis_col = best[1]; print(f"centre-of-rotation col {axis_col:.1f} (mirror NCC {best[0]:.4f}) of {W}")

    crop_w = min(largest_mult(2 * min(axis_col, W - axis_col)), largest_mult(W))
    crop_h = largest_mult(H)
    name = a.name or Path(a.mov).stem
    prof = DatasetProfile(name=name, mov_path=str(a.mov), fps=fps, height=H, width=W,
                          deg_per_frame=round(dpf, 6), rot_axis_col=round(axis_col, 2),
                          crop=(crop_h, crop_w), frame_start=a.frame_start, frame_end=a.frame_end,
                          memmap_path=None)
    out = a.out or str(Path(a.mov).parent / "tr_diffusion_profile.json")
    prof.save(out)
    print(f"\ncrop {(crop_h, crop_w)} (UNet-friendly, centred on axis)")
    print(f"wrote profile -> {out}\n{json.dumps(json.loads(Path(out).read_text()), indent=2)}")


if __name__ == "__main__":
    main()
