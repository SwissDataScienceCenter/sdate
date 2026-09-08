import json
from pathlib import Path

import h5py
import numpy as np

out_dir = Path("/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/denoised_v2_fullres_present_true/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01")
h5_path = out_dir / "SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
phase_path = out_dir / "SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
meta_path = out_dir / "denoise_metadata.json"

print("h5 size:", h5_path.stat().st_size / 1e9, "GB")
print("phase exists:", phase_path.exists(), "size:", phase_path.stat().st_size if phase_path.exists() else None)

with h5py.File(h5_path, "r") as f:
    d = f["exchange/data"]
    print("exchange/data shape/dtype:", d.shape, d.dtype)
    for i in [0, 4000, 10000, 16000, 19999]:
        frame = d[i]
        print(f"  frame {i}: min={frame.min()} max={frame.max()} mean={frame.mean():.2f}")
    print("exchange/data_dark shape/dtype:", f["exchange/data_dark"].shape, f["exchange/data_dark"].dtype)
    print("exchange/data_white shape/dtype:", f["exchange/data_white"].shape, f["exchange/data_white"].dtype)
    print("theta[:5]:", f["exchange/theta"][:5])

with open(meta_path) as fj:
    meta = json.load(fj)
print("--- metadata ---")
print(json.dumps(meta, indent=2))

orig_path = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
with h5py.File(orig_path, "r") as fo:
    orig0 = fo["exchange/data"][0]
    print("--- sanity: original frame 0 still raw/untouched ---")
    print("orig frame0 min/max/mean:", orig0.min(), orig0.max(), orig0.mean())
