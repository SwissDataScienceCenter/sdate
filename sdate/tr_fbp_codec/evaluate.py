"""Evaluate a trained checkpoint: real compression ratio via arithmetic coding
on held-out frames, compared against HEVC/FFV1 baselines.

Tiles each held-out frame into non-overlapping ``block_size`` blocks (128x512
divides evenly into 64x64 blocks -- 2x8 -- for the settled block_size=64;
adjust if block_size changes and it no longer divides evenly). Each block is
independently arithmetic-coded (one ``coding.encode`` call per block, batch
size 1, matching the paper's own independent-block design). Frame/context
edges are handled with edge-padding by ``block_margin`` -- note this is a
train/inference distribution mismatch AT THE TRUE IMAGE BOUNDARY specifically
(training never saw a padded crop, see data.py), a known minor edge effect,
not a correctness issue (arithmetic coding stays lossless regardless of how
well-calibrated the predicted distribution is).

Usage:
    python -m sdate.tr_fbp_codec.evaluate --ckpt /path/to/ckpt_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import baselines, coding
from .config import CodecConfig, DataConfig, QuantConfig
from .data import TrFbpCodecDataset
from .model import FramePredictorResNet3D, ModelConfig, stack_inputs
from .quantization import quantize_to_12bit


def _pad_hw(x: torch.Tensor, margin: int) -> torch.Tensor:
    """Edge-pad the last two (H, W) dims by ``margin`` on each side."""
    return torch.nn.functional.pad(x, (margin, margin, margin, margin), mode="replicate")


@torch.no_grad()
def evaluate_frame(model, ds: TrFbpCodecDataset, t: int, codec_cfg: CodecConfig,
                    quant_cfg: QuantConfig, device) -> dict:
    k = codec_cfg.k
    ctx_idx = np.arange(t - k, t)
    context = ds._R.native_window(ds.src, ctx_idx, ds.profile.crop, ds.profile.rot_axis_col)
    target_full = ds._R.native_window(ds.src, np.array([t]), ds.profile.crop, ds.profile.rot_axis_col)[0]
    prior_full = None
    if codec_cfg.use_fbp_prior:
        local = t - ds.tap_first
        prior_full = torch.from_numpy(np.asarray(ds.tap_mm[local]).astype(np.float32))

    bm, bs = codec_cfg.block_margin, codec_cfg.block_size
    h, w = target_full.shape
    assert h % bs == 0 and w % bs == 0, f"block_size={bs} must divide frame ({h},{w}) evenly"

    context_p = _pad_hw(context.unsqueeze(0), bm)[0]
    prior_p = _pad_hw(prior_full.unsqueeze(0), bm)[0] if prior_full is not None else None
    target_q_full = torch.from_numpy(quantize_to_12bit(target_full.numpy(), quant_cfg))

    total_bits, total_px = 0, 0
    for r0 in range(0, h, bs):
        for c0 in range(0, w, bs):
            ctx_blk = context_p[:, r0:r0 + bs + 2 * bm, c0:c0 + bs + 2 * bm].unsqueeze(0).to(device)
            prior_blk = None
            if prior_p is not None:
                prior_blk = prior_p[r0:r0 + bs + 2 * bm, c0:c0 + bs + 2 * bm].unsqueeze(0).to(device)
            target_blk = target_q_full[r0:r0 + bs, c0:c0 + bs].unsqueeze(0)

            x = stack_inputs(ctx_blk, prior_blk, n_levels=codec_cfg.n_classes)
            logits = model(x)[:, :, bm:bm + bs, bm:bm + bs].cpu()
            encoded = coding.encode(logits, target_blk)
            total_bits += len(encoded) * 8
            total_px += target_blk.numel()

    return {"frame": t, "bits": total_bits, "pixels": total_px, "bpp": total_bits / total_px}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--skip_baselines", action="store_true",
                    help="skip HEVC/FFV1 (needs ffmpeg, not present on every environment -- "
                         "e.g. the CSCS Clariden image). Baseline bpp is static per dataset, "
                         "so reuse an already-measured number rather than treating its absence "
                         "as 'no baseline exists'.")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = FramePredictorResNet3D(model_cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    ckpt_dir = Path(args.ckpt).parent
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    data_cfg = DataConfig(**cfg["data_cfg"])
    codec_cfg = CodecConfig(**cfg["codec_cfg"])
    quant_cfg = QuantConfig(**cfg["quant_cfg"])

    ds = TrFbpCodecDataset(data_cfg, codec_cfg, quant_cfg, split="holdout")
    targets = ds.targets
    if args.max_frames is not None:
        targets = targets[: args.max_frames]

    per_frame = []
    for t in targets:
        per_frame.append(evaluate_frame(model, ds, int(t), codec_cfg, quant_cfg, device))
        print(f"[eval] frame {t} bpp={per_frame[-1]['bpp']:.4f}")

    total_bits = sum(f["bits"] for f in per_frame)
    total_px = sum(f["pixels"] for f in per_frame)
    model_bpp = total_bits / total_px

    n_params = sum(p.numel() for p in model.parameters())
    weight_bits = n_params * 32  # fp32 state_dict, worst case (no weight compression)
    model_bpp_with_weights = (total_bits + weight_bits) / total_px

    result = {
        "n_frames": len(targets),
        "model_bpp_excl_weights": model_bpp,
        "model_bpp_incl_weights": model_bpp_with_weights,
        "n_model_params": n_params,
        "per_frame": per_frame,
    }

    if args.skip_baselines:
        print("[eval] --skip_baselines set: not recomputing HEVC/FFV1 (needs ffmpeg). "
              "Last measured on RunAI (200 held-out frames, same dataset): "
              "hevc12_bpp=6.266, ffv1_bpp=5.602. Also measured (sandbox, 2026-09-10, "
              "200- and 4000-frame holdout): ffv1_diff_bpp is WORSE than ffv1_bpp by "
              "~0.29 bpp (temporal-diff FFV1 hurts on this noisy data, see README.md "
              "'FFV1 on raw frames vs. temporal diff'). Compare against these, don't "
              "treat their absence here as 'no baseline'.")
    else:
        quant_arr = np.stack([
            quantize_to_12bit(
                ds._R.native_window(ds.src, np.array([int(t)]), ds.profile.crop, ds.profile.rot_axis_col)[0].numpy(),
                quant_cfg,
            ).astype(np.uint16)
            for t in targets
        ])
        fps = ds.profile.fps
        hevc_bytes = baselines.encode_hevc12_lossless(quant_arr, fps=fps)
        ffv1_bytes = baselines.encode_ffv1_lossless(quant_arr, fps=fps)
        ffv1_diff_bytes = baselines.encode_ffv1_lossless_diff(quant_arr, fps=fps)
        result["hevc12_bpp"] = baselines.bits_per_pixel(hevc_bytes, *quant_arr.shape)
        result["ffv1_bpp"] = baselines.bits_per_pixel(ffv1_bytes, *quant_arr.shape)
        result["ffv1_diff_bpp"] = baselines.bits_per_pixel(ffv1_diff_bytes, *quant_arr.shape)
    out_path = ckpt_dir / "eval_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame"}, indent=2))
    print(f"[eval] written -> {out_path}")


if __name__ == "__main__":
    main()
