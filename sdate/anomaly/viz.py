"""In-notebook visualisation: mask overlays, anomaly curves, interactive scrubber.

The scrubber is the "scroll and confirm" deliverable: a slider over turns shows
the fixed-angle frame with its anomaly mask overlaid on the left and the anomaly
curves (with control limit + a moving cursor) on the right, so you can check
that a curve excursion really coincides with visible structure in the frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _norm(img, vmin=None, vmax=None):
    img = np.asarray(img, dtype=np.float64)
    vmin = np.nanpercentile(img, 1) if vmin is None else vmin
    vmax = np.nanpercentile(img, 99) if vmax is None else vmax
    return np.clip((img - vmin) / (vmax - vmin + 1e-12), 0, 1)


def overlay_mask(frame, mask, cmap="hot", q=0.97, alpha=0.65, vmin=None, vmax=None):
    """RGB image: grayscale ``frame`` with the hot part of ``mask`` overlaid."""
    import matplotlib

    g = _norm(frame, vmin, vmax)
    rgb = np.repeat(g[..., None], 3, axis=2)
    m = np.abs(np.asarray(mask, dtype=np.float64))
    if m.max() > 0:
        thr = np.quantile(m, q)
        w = np.clip((m - thr) / (m.max() - thr + 1e-12), 0, 1) * (m >= thr)
        color = matplotlib.colormaps[cmap](_norm(m))[..., :3]
        rgb = (1 - alpha * w[..., None]) * rgb + (alpha * w[..., None]) * color
    return np.clip(rgb, 0, 1)


def plot_curves(ax, results, which="q", cursor_turn=None, ema=True):
    """Plot one angle's anomaly curve + control limit on ``ax``."""
    turns = results.turns
    y = (results.q_ema if ema else results.q) if which == "q" else (results.t2_ema if ema else results.t2)
    thr = results.q_thr if which == "q" else results.t2_thr
    flags = results.q_flags if which == "q" else results.t2_flags
    label = "Q / SPE" if which == "q" else "Hotelling T²"

    ax.plot(turns, y, lw=1.0, color="C0", label=label)
    thr_plot = np.where(np.isfinite(thr), thr, np.nan)  # inf during warm-up -> don't break autoscale
    ax.plot(turns, thr_plot, lw=1.0, ls="--", color="C3", label="control limit")
    if flags.any():
        ax.scatter(turns[flags], y[flags], s=14, color="C3", zorder=3, label="flagged")
    if cursor_turn is not None:
        ax.axvline(cursor_turn, color="k", lw=1.2, alpha=0.7)
    ax.set_xlabel("turn"); ax.set_ylabel(label); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"angle {results.angle:.1f}°  —  {label}")


def scrubber(detector, angle_index: int = 0, which: str = "q", figsize=(13, 4)):
    """Interactive slider over turns for one tracked angle (needs ipywidgets)."""
    import matplotlib.pyplot as plt
    from ipywidgets import IntSlider, interact

    res = detector.results[angle_index]
    if not res.frames:
        raise RuntimeError("No recorded frames — set DetectorConfig(record_frames=True).")
    frames = res.frame_stack()
    masks = res.mask_q_stack() if which == "q" else res.mask_t2_stack()
    turns = res.turns
    # cast off float16 first: np.nanpercentile overflows the percentile index in
    # float16 for large arrays and returns NaN (blanks the display).
    vmin, vmax = np.nanpercentile(frames.astype(np.float32), (1, 99))

    def _show(i):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw={"width_ratios": [1, 1.4]})
        axL.imshow(overlay_mask(frames[i], masks[i], vmin=vmin, vmax=vmax), aspect="auto")
        s = res.scores[i]
        tag = ("Q" if which == "q" else "T²") + (" ANOMALY" if (s.q_flag if which == "q" else s.t2_flag) else "")
        axL.set_title(f"turn {int(turns[i])}  (frame ~{s.x_pos:.0f})  {tag}")
        axL.set_xticks([]); axL.set_yticks([])
        plot_curves(axR, res, which=which, cursor_turn=turns[i])
        plt.tight_layout(); plt.show()

    interact(_show, i=IntSlider(min=0, max=len(turns) - 1, step=1, value=0, description="turn idx"))


def plot_all_angles(detector, which="q", figsize=(11, None)):
    """Stacked anomaly curves across all tracked angles (see majority voting)."""
    import matplotlib.pyplot as plt

    n = len(detector.results)
    h = figsize[1] or 2.2 * n
    fig, axs = plt.subplots(n, 1, figsize=(figsize[0], h), sharex=True)
    axs = np.atleast_1d(axs)
    for ai in range(n):
        plot_curves(axs[ai], detector.results[ai], which=which)
    plt.tight_layout()
    return fig
