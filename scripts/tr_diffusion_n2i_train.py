#!/usr/bin/env python3
"""Build a Noise2Inverse sub-reconstruction cache and train the N2I denoiser.

Self-supervised: no clean reference is used anywhere in training. A window's
projections are split into ``--k`` disjoint angular subsets, each reconstructed
independently, and the network learns to predict the held-out subset's
reconstruction from the mean of the others (the paper's ``X:1`` strategy).

    python scripts/tr_diffusion_n2i_train.py --k 4 --epochs 30 --tag k4
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/myhome/sdate"); sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R          # noqa: E402
from sdate.tr_diffusion.frames import MemmapFrameSource   # noqa: E402
from sdate.tr_diffusion.n2i import N2IUNet, reconstruct_splits, xk_pair  # noqa: E402

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
OUT = f"{CACHE}/n2i"
CK = "/myhome/sdate/checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_mm", default=f"{DATA}/denoised_212_Wunderkerze2_poissonhead_dose05.f16",
                   help="Denoised projection memmap whose reconstruction is being post-processed.")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--split_mode", choices=["interleaved", "block"], default="interleaved")
    p.add_argument("--block_size", type=int, default=1)
    p.add_argument("--n_windows", type=int, default=90, help="Windows to cache for training.")
    p.add_argument("--row_start", type=int, default=16)
    p.add_argument("--row_end", type=int, default=112)
    p.add_argument("--row_step", type=int, default=3)
    p.add_argument("--frame_start", type=int, default=400_300)
    p.add_argument("--frame_end", type=int, default=450_000)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--tag", default="k4")
    p.add_argument("--rebuild_cache", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    rows = np.arange(a.row_start, a.row_end, a.row_step)
    cache_path = f"{OUT}/subs_{a.tag}.f16"
    meta_path = cache_path + ".meta.json"

    # ---------------------------------------------------------------- cache
    if a.rebuild_cache or not Path(meta_path).exists():
        print("=== building sub-reconstruction cache ===", flush=True)
        dark = torch.from_numpy(np.load(f"{CACHE}/dark_map.npy")).float().to(dev)
        flat = torch.from_numpy(np.load(f"{CACHE}/flat_map.npy")).float().to(dev)
        src = MemmapFrameSource(f"{DATA}/frames_400k_500k.u16", f"{DATA}/212_Wunderkerze2.mov")
        m = np.load(a.source_mm + ".meta.npz")
        first, n = int(m["first_index"]), int(m["num_frames"])
        crop = (int(m["crop"][0]), int(m["crop"][1]))
        mm = np.memmap(a.source_mm, dtype=np.float16, mode="r", shape=(n, *crop))
        dpf = 1.801402
        win = R.window_length_frames(180.0, dpf)

        lo = max(a.frame_start, first)
        hi = min(a.frame_end, first + n)
        starts = [s for s in range(lo, hi, a.stride) if s + win <= hi]
        pick = [starts[i] for i in np.linspace(0, len(starts) - 1, min(a.n_windows, len(starts))).astype(int)]
        print(f"{len(starts)} windows available -> caching {len(pick)}", flush=True)

        arr = np.memmap(cache_path, dtype=np.float16, mode="w+",
                        shape=(len(pick), a.k, len(rows), crop[1], crop[1]))
        t0 = time.time()
        for i, s in enumerate(pick):
            idx = np.arange(s, s + win)
            ang = R.projection_angles(idx, deg_per_frame=dpf)
            counts = R.denoised_window_gpu(mm, first, idx, dev)
            at = R.destripe_sinogram(R.counts_to_attenuation_flatdark(counts, dark, flat), 31)
            subs = reconstruct_splits(at, ang, a.k, a.split_mode, a.block_size,
                                      det_bin=1, rows=rows, device=dev)
            arr[i] = subs.cpu().numpy().astype(np.float16)
            if i % 10 == 0:
                print(f"  cached {i+1}/{len(pick)}  ({time.time()-t0:.0f}s)", flush=True)
        arr.flush()
        scale = float(np.asarray(arr[: min(8, len(pick))]).astype(np.float32).std())
        json.dump(dict(window_starts=[int(x) for x in pick], k=a.k, rows=[int(r) for r in rows],
                       shape=list(arr.shape), scale=scale, split_mode=a.split_mode,
                       block_size=a.block_size, source_mm=a.source_mm),
                  open(meta_path, "w"), indent=2)
        print(f"cache written in {(time.time()-t0)/60:.1f} min -> {cache_path}", flush=True)
        del arr

    meta = json.load(open(meta_path))
    shape = tuple(meta["shape"])
    scale = float(meta["scale"])
    subs = np.memmap(cache_path, dtype=np.float16, mode="r", shape=shape)
    nW, k, nR = shape[0], shape[1], shape[2]
    print(f"cache {shape}  scale={scale:.6g}", flush=True)

    # ------------------------------------------------------------- dataset
    n_val = max(1, int(a.val_frac * nW))
    val_w = set(np.linspace(0, nW - 1, n_val).astype(int).tolist())
    train_w = [i for i in range(nW) if i not in val_w]
    val_w = sorted(val_w)
    print(f"train windows {len(train_w)}  val windows {len(val_w)}", flush=True)

    class DS(torch.utils.data.Dataset):
        def __init__(self, wins):
            self.items = [(w, r, j) for w in wins for r in range(nR) for j in range(k)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            w, r, j = self.items[i]
            s = torch.from_numpy(np.asarray(subs[w, :, r]).astype(np.float32))
            inp, tgt = xk_pair(s, j)
            return inp.unsqueeze(0) / scale, tgt.unsqueeze(0) / scale

    tl = torch.utils.data.DataLoader(DS(train_w), batch_size=a.batch_size, shuffle=True,
                                     num_workers=6, pin_memory=True, drop_last=True)
    vl = torch.utils.data.DataLoader(DS(val_w), batch_size=a.batch_size, shuffle=False, num_workers=4)

    model = N2IUNet(base=a.base).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(tl))
    scaler = torch.amp.GradScaler("cuda")
    ckpt = f"{CK}/n2i_{a.tag}.pt"
    best = float("inf")

    for ep in range(a.epochs):
        model.train(); tot = nb = 0; t0 = time.time()
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = nn.functional.mse_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        model.eval(); vt = vn = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                with torch.amp.autocast("cuda"):
                    vt += nn.functional.mse_loss(model(x), y).item()
                vn += 1
        tr, va = tot / max(nb, 1), vt / max(vn, 1)
        flag = ""
        if va < best:
            best = va
            torch.save(dict(model_state_dict=model.state_dict(), scale=scale, k=k,
                            base=a.base, meta=meta), ckpt)
            flag = "  *saved"
        print(f"epoch {ep:3d}  train {tr:.5f}  val {va:.5f}  ({time.time()-t0:.0f}s){flag}", flush=True)

    print(f"DONE best val {best:.5f} -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
