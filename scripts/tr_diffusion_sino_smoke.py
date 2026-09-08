#!/usr/bin/env python3
"""Quick GPU smoke test for sdate.tr_diffusion.sino_transform before committing
to a real training run: verifies the forward (Radon)/inverse (FBP) round trip
recovers a synthetic image, and that SinogramN2VLoss/denoise_frames_sinogram
run end to end on a tiny real batch from the actual dataset.

    python scripts/tr_diffusion_sino_smoke.py
"""
import sys

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

import numpy as np
import torch

from sdate.tr_diffusion.sino_transform import SinoTransform, choose_sino_shape, fit_sino_norm_range

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
assert device.type == "cuda", "this smoke test needs a real GPU (astra CUDA algorithms)"

vol_shape = (128, 512)
num_angles, det_cols = choose_sino_shape(vol_shape)
print("sino shape:", num_angles, det_cols)
st = SinoTransform(vol_shape, num_angles=num_angles, det_cols=det_cols, device=device)

# --- 1. forward/inverse round trip on a synthetic phantom ---
torch.manual_seed(0)
y, x = torch.meshgrid(torch.linspace(-1, 1, vol_shape[0], device=device),
                       torch.linspace(-1, 1, vol_shape[1], device=device), indexing="ij")
phantom = (200.0 + 300.0 * ((x - 0.1) ** 2 + (y * 4) ** 2 < 0.5).float()
                 + 150.0 * ((x + 0.3) ** 2 + (y * 4 - 0.2) ** 2 < 0.1).float())
phantom = phantom.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

sino = st.forward(phantom)
print("sino shape/range:", sino.shape, sino.min().item(), sino.max().item())
recon = st.inverse(sino)
err = (recon - phantom)
rmse = err.pow(2).mean().sqrt().item()
rng = phantom.max().item() - phantom.min().item()
psnr = 20 * np.log10(rng / max(rmse, 1e-6))
print(f"round-trip RMSE={rmse:.3f}  PSNR={psnr:.2f} dB (expect high, i.e. FBP correctly inverts the forward transform)")
assert psnr > 25, f"round-trip PSNR too low ({psnr:.2f} dB) -- geometry/padding likely wrong"

# --- 2. real data: dataset -> SinogramN2VLoss -> model, one step ---
sys.path.insert(0, "/myhome/BaseTraining")
from sdate.tr_diffusion.data import TimeResolvedFrameDataset
from sdate.tr_diffusion.losses import SinogramN2VLoss
from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.pipeline import denoise_frames_sinogram

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
ds = TimeResolvedFrameDataset(
    mov_path=f"{DATA}/212_Wunderkerze2.mov", memmap_path=f"{DATA}/frames_400k_500k.u16",
    k=1, frame_start=400_000, frame_end=400_500, crop=(128, 512),
    extra_noise_dose=0.05, noise_seed=12345,
)
print("dataset norm_min/max:", ds.norm_min, ds.norm_max, "in_channels_baseline:", ds.in_channels_baseline)

sino_norm_min, sino_norm_max = fit_sino_norm_range(ds, st, n_sample=8, seed=0)
print("sino_norm:", sino_norm_min, sino_norm_max)

model = create_baseline_unet(k=1, sample_size=(num_angles, det_cols), poisson_head=False).to(device)
loss_fn = SinogramN2VLoss(device, st, norm_min=ds.norm_min, norm_max=ds.norm_max,
                          sino_norm_min=sino_norm_min, sino_norm_max=sino_norm_max)

loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
batch = next(iter(loader))
loss, stats = loss_fn.compute_loss(batch, model)
print("loss:", stats)
loss.backward()
grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print("grad_norm (should be > 0):", grad_norm)
assert grad_norm > 0

# --- 3. inference path ---
model.eval()
with torch.no_grad():
    central = batch["central"].to(device)
    context = batch["context"].to(device)
    den = denoise_frames_sinogram(model, st, central, context,
                                  norm_min=ds.norm_min, norm_max=ds.norm_max,
                                  sino_norm_min=sino_norm_min, sino_norm_max=sino_norm_max)
print("denoise_frames_sinogram output shape:", den.shape)
assert den.shape == central.shape

print("ALL SMOKE CHECKS PASSED")
