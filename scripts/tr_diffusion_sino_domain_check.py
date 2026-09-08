#!/usr/bin/env python3
"""Visualize the sinogram domain itself (clean / noisy / denoised) for a few
example frames -- the intermediate representation SinogramN2VLoss actually
trains/denoises in, before the final inverse-FBP step back to projection space
(see tr_diffusion_sino_projection_check.py for that projection-domain view).

    python scripts/tr_diffusion_sino_domain_check.py
"""
import json
import os
import sys

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")
sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/BaseTraining")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sdate.tr_diffusion.data import TimeResolvedFrameDataset
from sdate.tr_diffusion.load import load_denoiser
from sdate.tr_diffusion.n2v import blind_spot_corrupt
from sdate.tr_diffusion.sino_transform import SinoTransform

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CKPT = "/myhome/data/sdate/shared/checkpoints/tr_denoise_sinogram_k1dose005_v1.pt"
MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"
DOSE, NOISE_SEED = 0.05, 12345
FRAME_START, FRAME_END = 420_000, 420_900
OUT_PNG = "/myhome/sdate/scripts/sino_domain_check.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

model, cfg = load_denoiser(CKPT, device=device)
model.eval()
lo_n, hi_n = float(cfg["norm_min"]), float(cfg["norm_max"])
sino_norm_min, sino_norm_max = float(cfg["sino_norm_min"]), float(cfg["sino_norm_max"])
crop = tuple(cfg["crop"])
present = float(cfg.get("conditioning_probability", 1.0)) > 0.0

sino = SinoTransform(crop, num_angles=int(cfg["sino_num_angles"]), det_cols=int(cfg["sino_det_cols"]),
                     angle_max_deg=float(cfg["sino_angle_max_deg"]), device=device)
print(f"sino shape=({sino.num_angles},{sino.det_cols})", flush=True)

ds = TimeResolvedFrameDataset(
    mov_path=MOV, memmap_path=MEMMAP, k=int(cfg["k"]),
    frame_start=FRAME_START, frame_end=FRAME_END, crop=crop,
    neighborhoods=cfg.get("neighborhoods", "both"),
    include_mirror=bool(cfg.get("include_mirror", False)),
    temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", False)),
    norm_range=(lo_n, hi_n), extra_noise_dose=DOSE, noise_seed=NOISE_SEED,
)
first_idx = int(ds.indices.min())
idx_by_frame = {int(fi): i for i, fi in enumerate(ds.indices)}
example_frames = [first_idx + 10, first_idx + 150, first_idx + 300, first_idx + 480]


def denorm(x, lo, hi):
    return (x.float() + 1.0) * 0.5 * (hi - lo) + lo


def sino_norm(s, lo, hi):
    return 2.0 * (s - lo) / (hi - lo) - 1.0


def sino_denorm(s, lo, hi):
    return (s.float() + 1.0) * 0.5 * (hi - lo) + lo


fig, axes = plt.subplots(len(example_frames), 3, figsize=(15, 4.2 * len(example_frames)))

with torch.no_grad():
    for row, f in enumerate(example_frames):
        item = ds[idx_by_frame[f]]
        central = item["central"].unsqueeze(0).to(device)
        context = item["context"].unsqueeze(0).to(device)
        reference = item["reference"].unsqueeze(0).to(device)

        central_counts = denorm(central, lo_n, hi_n)
        context_counts = denorm(context, lo_n, hi_n)
        reference_counts = denorm(reference, lo_n, hi_n)

        clean_sino = sino.forward(reference_counts)[0, 0].cpu().numpy()
        noisy_sino = sino.forward(central_counts)[0, 0].cpu().numpy()

        sino_central_n = sino_norm(sino.forward(central_counts), sino_norm_min, sino_norm_max)
        sino_context_n = (sino_norm(sino.forward(context_counts), sino_norm_min, sino_norm_max)
                          if context_counts.shape[1] > 0 else sino.forward(context_counts))
        corrupted = blind_spot_corrupt(sino_central_n)[0] if present else torch.zeros_like(sino_central_n)
        model_input = torch.cat([corrupted, sino_context_n], dim=1)
        t = torch.zeros(1, device=device, dtype=torch.long)
        cls = torch.full((1,), int(present), device=device, dtype=torch.long)
        pred_sino_n = model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]
        denoised_sino = sino_denorm(pred_sino_n, sino_norm_min, sino_norm_max)[0, 0].cpu().numpy()

        vmin, vmax = np.percentile(clean_sino, [1, 99])
        for col, (name, img) in enumerate([("clean sinogram", clean_sino), ("noisy sinogram", noisy_sino),
                                           ("denoised sinogram", denoised_sino)]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"frame {f}: {name}", fontsize=9)
            ax.axis("off")
        print(f"frame {f}: clean mean={clean_sino.mean():.1f} noisy mean={noisy_sino.mean():.1f} "
              f"denoised mean={denoised_sino.mean():.1f}", flush=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=110)
print("saved", OUT_PNG, flush=True)
print("DONE", flush=True)
