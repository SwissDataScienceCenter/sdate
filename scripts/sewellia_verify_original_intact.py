import hashlib
from pathlib import Path

p = Path("/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5")
st = p.stat()
print("size", st.st_size, "mtime", st.st_mtime)

import h5py
import numpy as np
with h5py.File(p, "r") as f:
    d = f["exchange/data"]
    print("shape", d.shape, "dtype", d.dtype)
    # sample a few frames and check they look like raw (unmodified) counts, not denoised floats-rounded
    s = d[0]
    print("frame0 min/max/mean", s.min(), s.max(), s.mean())
    s2 = d[10000]
    print("frame10000 min/max/mean", s2.min(), s2.max(), s2.mean())
