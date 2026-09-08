"""Export a shareable ``.mp4``: fixed-angle movie + mask overlay + synced curve.

Each output frame is a matplotlib composite — the fixed-angle projection with
its anomaly mask overlaid on the left, and the anomaly curve with a moving
cursor + control limit on the right — rendered to RGB and piped to ffmpeg.  This
mirrors ``notebooks/wunderkerze_rotation_cache/make_movies.py`` but adds the
anomaly layer, giving the "movie with mask + curve next to it" deliverable in a
form that needs no live kernel.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .viz import overlay_mask, plot_curves

FFMPEG = "/myhome/bin/ffmpeg"


def export_overlay_movie(detector, out_path, angle_index: int = 0, which: str = "q",
                         fps: int = 12, stride: int = 1, dpi: int = 100,
                         ffmpeg: str = FFMPEG, crf: int = 18,
                         figsize=(12, 4)) -> Path:
    """Render an anomaly-overlay movie for one tracked angle to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    res = detector.results[angle_index]
    if not res.frames:
        raise RuntimeError("No recorded frames — set DetectorConfig(record_frames=True).")
    frames = res.frame_stack()
    masks = res.mask_q_stack() if which == "q" else res.mask_t2_stack()
    turns = res.turns
    idxs = range(0, len(turns), stride)
    # cast off float16 first (np.nanpercentile overflows -> NaN on large float16 arrays)
    vmin, vmax = np.nanpercentile(frames.astype(np.float32), (1, 99))

    out_path = Path(out_path)
    proc = None
    W = H = None
    for i in idxs:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw={"width_ratios": [1, 1.4]}, dpi=dpi)
        # Pin the layout so every rendered frame has *identical* pixel dimensions
        # (ffmpeg rawvideo needs a constant frame size); a title that grows on
        # anomaly frames must not reflow the axes.
        fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.12, wspace=0.18)
        axL.imshow(overlay_mask(frames[i], masks[i], vmin=vmin, vmax=vmax), aspect="auto")
        s = res.scores[i]
        flagged = (s.q_flag if which == "q" else s.t2_flag)
        axL.set_title(f"angle {res.angle:.1f}°  turn {int(turns[i])}"
                      + ("   ANOMALY" if flagged else ""))
        axL.set_xticks([]); axL.set_yticks([])
        plot_curves(axR, res, which=which, cursor_turn=turns[i])
        # Grab pixels exactly the way savefig does (full Agg render), not
        # buffer_rgba after a bare draw() — the latter can drop the imshow layer.
        buf, (w, h) = fig.canvas.print_to_buffer()
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 4)[..., :3]
        plt.close(fig)

        if proc is None:
            H, W = rgb.shape[:2]
            proc = subprocess.Popen(
                [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0",
                 "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(out_path)],
                stdin=subprocess.PIPE)
        if rgb.shape[:2] != (H, W):               # safety: keep frame size constant
            rgb = rgb[:H, :W]
        proc.stdin.write(np.ascontiguousarray(rgb).tobytes())

    if proc is not None:
        proc.stdin.close(); proc.wait()
    return out_path
