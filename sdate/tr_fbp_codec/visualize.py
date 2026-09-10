"""Visual sanity check of one training sample: context + prior tap vs. target.

Renders a PNG with the model's actual input stack (``k`` context frames,
oldest-first, plus the FBP-prior tap when enabled -- same channel order as
``model.stack_inputs``) alongside the target, plus the per-channel
difference against the target so context/prior quality is inspectable by
eye, not just via the aggregate bpp number.

    python -m sdate.tr_fbp_codec.visualize --out /path/to/context_check.png
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import CodecConfig, DataConfig, QuantConfig
from .data import TrFbpCodecDataset


def make_context_figure(sample: dict, block_margin: int, block_size: int, out_path: str) -> None:
    import matplotlib.pyplot as plt

    context = sample["context"].numpy()  # (k, H, W), oldest-first
    prior = sample.get("prior")
    prior = prior.numpy() if prior is not None else None
    target_q = sample["target_q"].numpy()  # (block_size, block_size), the supervised interior
    bm, bs = block_margin, block_size

    def interior(full: np.ndarray) -> np.ndarray:
        if bs is None:  # full-frame mode: no margin crop applied by the dataset
            return full
        return full[bm:bm + bs, bm:bm + bs]

    k = context.shape[0]
    panels = [interior(context[i]) for i in range(k)]
    labels = [f"context t-{k - i}" for i in range(k)]
    if prior is not None:
        panels.append(interior(prior))
        labels.append("prior (FBP)")
    panels.append(target_q)
    labels.append("target")

    diffs = [target_q - p for p in panels[:-1]]  # skip the target-vs-itself panel
    diff_labels = [f"target - {lbl}" for lbl in labels[:-1]]

    n_cols = len(panels)
    vmin = min(p.min() for p in panels)
    vmax = max(p.max() for p in panels)
    dmax = max(np.abs(d).max() for d in diffs) if diffs else 1.0

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6.5))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_cols):
        ax = axes[0, col]
        im = ax.imshow(panels[col], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(labels[col], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.7, label="raw counts (12-bit)")

    for col in range(n_cols - 1):
        ax = axes[1, col]
        imd = ax.imshow(diffs[col], cmap="seismic", vmin=-dmax, vmax=dmax)
        ax.set_title(diff_labels[col], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, n_cols - 1].axis("off")
    fig.colorbar(imd, ax=axes[1, :-1].tolist(), shrink=0.7, label="signed diff")

    fig.suptitle(f"frame {sample['frame_idx']} -- model input context vs. target")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--use_fbp_prior", type=int, default=None, choices=[0, 1])
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--split", type=str, default="train", choices=["train", "holdout"])
    p.add_argument("--frame_idx", type=int, default=None, help="specific target frame; default random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    data_cfg = DataConfig()
    codec_cfg = CodecConfig()
    if args.k is not None:
        codec_cfg.k = args.k
    if args.use_fbp_prior is not None:
        codec_cfg.use_fbp_prior = bool(args.use_fbp_prior)
    if args.block_size is not None:
        codec_cfg.block_size = args.block_size
    quant_cfg = QuantConfig(mode="truncate")

    ds = TrFbpCodecDataset(data_cfg, codec_cfg, quant_cfg, split=args.split, crops_per_frame=1)

    np.random.seed(args.seed)
    if args.frame_idx is not None:
        matches = np.where(ds.targets == args.frame_idx)[0]
        if len(matches) == 0:
            raise ValueError(f"frame {args.frame_idx} not in split={args.split!r} target range")
        idx = int(matches[0])
    else:
        idx = int(np.random.randint(0, len(ds)))

    sample = ds[idx]
    make_context_figure(sample, ds.block_margin, ds.block_size, args.out)
    print(f"[visualize] frame={sample['frame_idx']} -> {args.out}")


if __name__ == "__main__":
    main()
