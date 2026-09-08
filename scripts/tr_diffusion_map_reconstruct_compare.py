#!/usr/bin/env python3
"""Compare Gamma-Poisson MAP reconstruction (sdate.tr_diffusion.map_reconstruct)
against plain FBP arms, on real wunderkerze2 dose-0.05 data, using the
poisson_head denoiser's own (mu_net, var_net) belief as the MAP prior.

Arms (identical geometry, circular-masked PSNR/SSIM/sharpness vs GT):
  GT          -- FBP of the native low-noise measurement
  noisy       -- FBP of the raw dose-0.05 measurement (no denoising)
  baseline    -- FBP of the plain-Huber denoiser's point estimate
  poissonhead -- FBP of the poisson_head denoiser's posterior-mean point estimate
  map         -- map_reconstruct's Gamma-Poisson MAP volume, built from the SAME
                 raw dose-0.05 counts + poisson_head's PRIOR (mu_net, var_net)
                 -- the network's belief BEFORE posterior combination -- never
                 from a denoised point estimate (see map_reconstruct.py docstring).

baseline/poissonhead arms reuse the ALREADY-CACHED denoised memmaps from the
earlier ablation pipelines (no need to re-run inference for those). Only the
MAP prior (mu_net, var_net) requires a fresh forward pass, since the cached
poissonhead memmap only holds the POSTERIOR-COMBINED result, not the raw
two-head belief.

Runs on a handful of representative windows only -- MAP/GD is far more
expensive per-window than a single FBP call -- at det_bin=2 for tractable
GD cost. Needs a real GPU + ASTRA (astra_torch): submit via RunAI, e.g.

  runai workspace submit sdate-map-recon-cmp \\
    -i lfbarba/sdsc_image:1.0.1 -p sdate-luisb \\
    --gpu-request-type portion --gpu-portion-request 0.2 \\
    --node-type A100 --large-shm --cpu-core-request 4 --cpu-core-limit 10 \\
    --cpu-memory-limit 64G --preemptibility preemptible \\
    --command -- bash -c "cd /myhome/sdate && python -m pip install -e . -q && \\
      python scripts/tr_diffusion_map_reconstruct_compare.py"
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from torch.utils.data import DataLoader, Subset  # noqa: E402

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser  # noqa: E402
from sdate.tr_diffusion.map_reconstruct import map_reconstruct  # noqa: E402
from sdate.tr_diffusion.n2v import blind_spot_corrupt  # noqa: E402
from sdate.tr_diffusion.nb_head import split_mu_var  # noqa: E402

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
CK = "/myhome/sdate/checkpoints"

MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"
BASELINE_MM = f"{DATA}/denoised_baseline_dose005.f16"
POISSONHEAD_MM = f"{DATA}/denoised_212_Wunderkerze2_poissonhead_dose05.f16"
POISSONHEAD_CKPT = f"{CK}/tr_denoise_baseline_k1_dose005_poissonhead.pt"

DOSE, NOISE_SEED = 0.05, 12345
DET_BIN = 2
N_ITERS, LR = 200, 1e-1
CROP = (128, 512)
AXIS_COL = R.ROT_AXIS_COL

# Start small (3 windows) -- MAP/GD is expensive and this hasn't run on real
# data before; widen WINDOW_STARTS once this is confirmed working.
WINDOW_STARTS = [410300, 425300, 440300]

OUT_DIR = Path(CACHE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def get_map_prior(frame_indices: np.ndarray):
    """(mu_net, var_net) -- (V,R,C) raw-count-units two-head belief -- for
    EXACTLY these frame numbers, in that order, from the poisson_head model.
    """
    model, cfg = load_denoiser(POISSONHEAD_CKPT, device=device)
    assert cfg.get("poisson_head"), f"{POISSONHEAD_CKPT} is not a poisson_head checkpoint"
    # temporal_raw_pairs context taps reach ~PERIOD_360 (~200 frames) away --
    # usable_frame_range needs > 2*margin frames of headroom around the window
    # (see geometry.py) or it raises; +5 alone (a window's-width margin) is not
    # nearly enough and was caught here before submission by exactly that error.
    margin = 220
    lo_f, hi_f = int(frame_indices.min()) - margin, int(frame_indices.max()) + margin
    ds = TimeResolvedFrameDataset(
        mov_path=MOV, memmap_path=MEMMAP, k=int(cfg["k"]),
        frame_start=lo_f, frame_end=hi_f, crop=tuple(cfg["crop"]),
        neighborhoods=cfg.get("neighborhoods", "both"),
        norm_range=(float(cfg["norm_min"]), float(cfg["norm_max"])),
        extra_noise_dose=DOSE, noise_seed=NOISE_SEED,
        temporal_raw_pairs=bool(cfg.get("temporal_raw_pairs", False)),
        cond_frame_start=cfg.get("frame_start"), cond_frame_end=cfg.get("frame_end"),
    )
    pos = np.searchsorted(ds.indices, frame_indices)
    assert np.array_equal(ds.indices[pos], frame_indices), "requested frames not all usable by this checkpoint"

    # shuffle=False -> DataLoader yields items in exactly Subset(ds, pos)'s order,
    # i.e. exactly frame_indices' order, regardless of num_workers -- so no
    # reordering is needed; the frame_index assertion below just confirms that
    # guarantee held rather than silently trusting it.
    loader = DataLoader(Subset(ds, pos.tolist()), batch_size=32, shuffle=False, num_workers=0)
    mus, vars_, order = [], [], []
    with torch.no_grad():
        for item in loader:
            central = item["central"].to(device)
            context = item["context"].to(device)
            corrupted, _ = blind_spot_corrupt(central)
            model_input = torch.cat([corrupted, context], dim=1)
            t = torch.zeros(central.shape[0], dtype=torch.long, device=device)
            cls = torch.ones(central.shape[0], dtype=torch.long, device=device)
            raw = model(model_input, timestep=t, class_labels=cls, return_dict=False)[0]
            mu, var = split_mu_var(raw)  # already raw-count units, no denorm needed
            mus.append(mu[:, 0].cpu())
            vars_.append(var[:, 0].cpu())
            order.append(item["frame_index"].numpy())
    mu_all = torch.cat(mus, dim=0)
    var_all = torch.cat(vars_, dim=0)
    order = np.concatenate(order)
    assert np.array_equal(order, frame_indices), "DataLoader did not preserve frame order as expected"
    return mu_all.to(device), var_all.to(device)


def process_window(src, s: int, dark, flat, mask, log_every_map: int = 50) -> dict:
    win = R.window_length_frames(180.0)
    idx = np.arange(s, s + win)
    ang = R.projection_angles(idx, deg_per_frame=R.DEG_PER_FRAME)

    gt = R.native_window_gpu(src, idx, CROP, AXIS_COL, device)
    g = torch.Generator(device=device).manual_seed(NOISE_SEED + int(s))
    noisy = R.noisy_window_gpu(gt, DOSE, generator=g)

    bmm = np.load(BASELINE_MM + ".meta.npz")
    baseline_counts = R.denoised_window_gpu(
        np.memmap(BASELINE_MM, dtype=np.float16, mode="r", shape=(int(bmm["num_frames"]), *CROP)),
        int(bmm["first_index"]), idx, device)
    phmm = np.load(POISSONHEAD_MM + ".meta.npz")
    poissonhead_counts = R.denoised_window_gpu(
        np.memmap(POISSONHEAD_MM, dtype=np.float16, mode="r", shape=(int(phmm["num_frames"]), *CROP)),
        int(phmm["first_index"]), idx, device)

    log(f"  window {s}: fetching MAP prior (mu_net, var_net) for {len(idx)} frames")
    mu_net, var_net = get_map_prior(idx)

    counts = {"GT": gt, "noisy": noisy, "baseline": baseline_counts, "poissonhead": poissonhead_counts}
    atten = {a: R.counts_to_attenuation_flatdark(counts[a], dark, flat) for a in counts}
    atten = {a: R.destripe_sinogram(atten[a], 31) for a in atten}
    vols = {a: R.reconstruct(atten[a], ang, det_bin=DET_BIN, method="fbp", device=device) for a in atten}

    dark_b = R.bin_detector(dark.unsqueeze(0), DET_BIN)[0]
    flat_b = R.bin_detector(flat.unsqueeze(0), DET_BIN)[0]
    noisy_b = R.bin_detector(noisy, DET_BIN)
    mu_net_b = R.bin_detector(mu_net, DET_BIN)
    var_net_b = R.bin_detector(var_net, DET_BIN) / float(DET_BIN * DET_BIN)  # binned mean's variance shrinks by N

    log(f"  window {s}: running map_reconstruct ({N_ITERS} iters)")
    t0 = time.time()
    vols["map"] = map_reconstruct(
        noisy_b, mu_net_b, var_net_b, ang, flat_b, dark_b,
        vol_shape=None, n_iters=N_ITERS, lr=LR, warm_start="fbp",
        dose=DOSE, log_every=log_every_map, device=device,
    )
    log(f"  window {s}: map_reconstruct done in {(time.time() - t0):.1f}s")

    gtv = vols["GT"]
    dr = float(gtv[..., mask].max() - gtv[..., mask].min())
    metrics = {}
    for a in ("noisy", "baseline", "poissonhead", "map"):
        ps, ss = R.masked_scores(gtv, vols[a], mask, dr)
        sh = R.masked_sharpness(gtv, vols[a], mask)
        metrics[a] = {"psnr": float(ps), "ssim": float(ss), "sharpness": float(sh)}
        log(f"    {a:12s} PSNR {ps:6.2f}  SSIM {ss:.3f}  sharpness {sh:.3f}")

    mid = vols["GT"].shape[0] // 2
    slices = {a: vols[a][mid].detach().cpu() for a in vols}
    return {"window_start": s, "metrics": metrics, "slices": slices}


def main():
    log(f"device={device}  windows={WINDOW_STARTS}  det_bin={DET_BIN}  n_iters={N_ITERS} lr={LR}")
    src = R.MemmapFrameSource(MEMMAP, MOV)
    dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float().to(device)
    flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float().to(device)
    nslices, hplane = CROP[0] // DET_BIN, CROP[1] // DET_BIN
    mask = R.make_mask(hplane, hplane).to(device)

    results = []
    for s in WINDOW_STARTS:
        log(f"=== window start {s} ===")
        results.append(process_window(src, s, dark, flat, mask))

    summary = {"windows": WINDOW_STARTS, "det_bin": DET_BIN, "n_iters": N_ITERS, "lr": LR, "dose": DOSE}
    for arm in ("noisy", "baseline", "poissonhead", "map"):
        for metric in ("psnr", "ssim", "sharpness"):
            vals = [r["metrics"][arm][metric] for r in results]
            summary[f"{arm}_{metric}_mean"] = float(np.mean(vals))
    (OUT_DIR / "recon_summary_map_vs_baseline.json").write_text(json.dumps(summary, indent=2))
    log("SUMMARY " + json.dumps(summary, indent=2))

    vmin, vmax = np.percentile(results[0]["slices"]["GT"].numpy(), [1, 99])
    arms_order = ["GT", "noisy", "baseline", "poissonhead", "map"]
    combined = [torch.cat([r["slices"][a] for a in arms_order], dim=1) for r in results]
    R.write_slice_movie(combined, OUT_DIR / "recon_map_vs_baseline.mov", float(vmin), float(vmax))
    log(f"movie written -> {OUT_DIR / 'recon_map_vs_baseline.mov'}  panels: {' | '.join(arms_order)}")
    log("DONE")


if __name__ == "__main__":
    main()
