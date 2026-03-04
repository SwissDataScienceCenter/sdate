#!/usr/bin/env python3
"""
projections_to_sinograms.py

Converts a stack of projection TIFF files into a sinogram volume saved as HDF5,
without loading the full volume into memory.

Layout
------
  Input  : N_proj projection images, each of shape (H, W)
  Output : HDF5 dataset "sinograms" of shape (H, N_proj, W)
           sinograms[h, :, :] is the sinogram at height h
             → rows are angle index (projection), columns are detector pixel

Algorithm (chunk-wise, memory-bounded)
---------------------------------------
  For each row-chunk  [h_start, h_end)  of size k:
    1. Preallocate buffer : (N_proj, k, W)   ← one entry per projection
    2. Loop over all N_proj projections:
         proj  = processor.get_projection(proj_idx, normalize=False)   # (H, W)
         buffer[proj_idx, :, :] = proj[h_start:h_end, :]
    3. Transpose buffer to (k, N_proj, W)   ← sinogram layout
    4. Write to HDF5:  sinograms[h_start:h_end, :, :]  = transposed buffer
    5. Free buffer, advance to next chunk

Memory footprint per chunk (k=100, N_proj=1501, W=2560, float32):
    100 × 1501 × 2560 × 4 B ≈ 1.5 GB
"""


import argparse
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so `sdate` is importable
# when running the script directly (python scripts/projections_to_sinograms.py)
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import h5py

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:  # minimal fallback
        def __init__(self, iterable=None, **kw):
            self._it = iterable
            self._total = kw.get("total", None)
            self._desc = kw.get("desc", "")
            self._n = 0
        def __iter__(self):
            for item in self._it:
                yield item
                self._n += 1
        def update(self, n=1):
            self._n += n
        def __enter__(self): return self
        def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def build_processor(folder: Path, verbose: bool, use_attenuation: bool = False):
    try:
        from sdate.datasets.projection_triplet_dataset import TomographyFolderProcessor
    except ImportError as e:
        print(f"❌ Could not import TomographyFolderProcessor: {e}")
        print("   Make sure sdate is installed / on PYTHONPATH.")
        sys.exit(1)

    return TomographyFolderProcessor(
        folder_path=folder,
        num_darks=None,    # auto-detect from log file
        num_flats=None,    # auto-detect from log file
        cache_in_memory=False,
        verbose=verbose,
        use_attenuation=use_attenuation,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert projection TIFFs to a sinogram HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to the tomography folder (must contain a .log file for auto-detection).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output HDF5 path. Defaults to <folder>/sinograms.h5",
    )
    parser.add_argument(
        "--chunk-rows", "-k",
        type=int,
        default=100,
        dest="chunk_rows",
        help="Number of height rows to process per pass (controls RAM usage).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "uint16"],
        help=(
            "Storage dtype in HDF5. "
            "float32 = lossless (pixel values as loaded). "
            "uint16   = cast back to 16-bit integers (saves ~50%% space, lossless for raw TIFF). "
            "float16  = lossy half-precision."
        ),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Enable LZF compression in HDF5 (adds chunking; slower write, smaller file).",
    )
    parser.add_argument(
        "--attenuation",
        action="store_true",
        default=False,
        help=(
            "Compute and save attenuation projections instead of raw pixel values. "
            "Loads dark and flat fields, then applies: -log((proj - dark) / (flat - dark + ε)). "
            "Incompatible with --dtype uint16."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress bars.",
    )
    args = parser.parse_args()

    if args.attenuation and args.dtype == "uint16":
        print("❌ --attenuation is incompatible with --dtype uint16 (attenuation values can be negative).")
        print("   Use --dtype float32 (default) or --dtype float16.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # resolve paths
    # ------------------------------------------------------------------
    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve() if args.output else folder / "sinograms.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # ------------------------------------------------------------------
    # initialise processor  (reads log file, discovers darks/flats/projs)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n📂 Initialising TomographyFolderProcessor …")
    processor = build_processor(folder, verbose=verbose, use_attenuation=args.attenuation)

    N_proj = processor.num_projections
    H      = processor.original_height
    W      = processor.original_width
    k      = args.chunk_rows
    n_chunks = (H + k - 1) // k
    dtype  = np.dtype(args.dtype)

    buf_bytes = k * N_proj * W * np.dtype("float32").itemsize
    out_bytes = H * N_proj * W * dtype.itemsize

    if verbose:
        print(f"\n📊 Dataset summary")
        print(f"   Projections : {N_proj}")
        print(f"   Height (H)  : {H}")
        print(f"   Width  (W)  : {W}")
        print(f"   Sinogram volume shape : ({H}, {N_proj}, {W})  [{args.dtype}]")
        print(f"   Estimated output size : {fmt_bytes(out_bytes)}")
        print(f"   Chunk rows  : {k}  →  {n_chunks} chunks")
        print(f"   Buffer/chunk: {fmt_bytes(buf_bytes)}")
        print(f"   Output      : {output_path}")
        print(f"   Compression : {'LZF' if args.compress else 'none (contiguous)'}")
        print(f"   Mode        : {'attenuation  (-log((I-dark)/(flat-dark+ε)))' if args.attenuation else 'raw projections'}")

    # ------------------------------------------------------------------
    # pre-load darks & flats when attenuation mode is requested so that
    # the overhead is paid once, before the timed chunk loop starts
    # ------------------------------------------------------------------
    if args.attenuation:
        if verbose:
            print(f"\n🔬 Pre-loading dark and flat fields …")
        processor._load_darks_flats()

    # ------------------------------------------------------------------
    # create HDF5 file & preallocate dataset
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n💾 Creating HDF5 file …")

    hdf5_opts: dict = {}
    if args.compress:
        # chunk one sinogram row at a time → optimal for sinogram reads
        hdf5_opts["compression"] = "lzf"
        hdf5_opts["chunks"]      = (1, N_proj, W)
    # else: contiguous layout – a single sinogram ds[h] is one contiguous read

    t_start = time.time()

    with h5py.File(output_path, "w") as f:
        f.attrs["source_folder"]  = str(folder)
        f.attrs["num_projections"] = N_proj
        f.attrs["height"]         = H
        f.attrs["width"]          = W
        f.attrs["sinogram_shape"] = f"(H={H}, N_proj={N_proj}, W={W})"
        f.attrs["attenuation"]    = args.attenuation
        f.attrs["description"]    = (
            "Sinogram volume. "
            "sinograms[h, :, :] → sinogram at detector row h; "
            "axis-0 = height, axis-1 = projection / angle, axis-2 = detector column. "
            + (
                "Values are attenuation projections: -log((I - dark) / (flat - dark + ε))."
                if args.attenuation else
                "Values are raw pixel intensities."
            )
        )

        ds = f.create_dataset(
            "sinograms",
            shape=(H, N_proj, W),
            dtype=dtype,
            **hdf5_opts,
        )

        # ----------------------------------------------------------------
        # chunk loop
        # ----------------------------------------------------------------
        for chunk_idx in range(n_chunks):
            h_start  = chunk_idx * k
            h_end    = min(h_start + k, H)
            actual_k = h_end - h_start

            if verbose:
                elapsed = time.time() - t_start
                print(
                    f"\n🔄 Chunk {chunk_idx + 1}/{n_chunks}  "
                    f"rows {h_start}–{h_end - 1}  "
                    f"(elapsed {elapsed:.0f}s)"
                )

            # ----------------------------------------------------------
            # 1. preallocate buffer  (N_proj, actual_k, W)  – float32
            # ----------------------------------------------------------
            buffer = np.empty((N_proj, actual_k, W), dtype=np.float32)

            # ----------------------------------------------------------
            # 2. fill buffer: one projection at a time
            # ----------------------------------------------------------
            proj_iter = range(N_proj)
            if HAS_TQDM and not args.quiet:
                proj_iter = tqdm(proj_iter, desc="  loading projections", unit="proj", leave=False)

            for proj_idx in proj_iter:
                # normalize=False: return raw or attenuation values (no [0,1] rescaling)
                # attenuation is computed inside the processor when use_attenuation=True
                proj = processor.get_projection(proj_idx, normalize=False)  # (H, W) float32
                buffer[proj_idx, :, :] = proj[h_start:h_end, :]

            # ----------------------------------------------------------
            # 3. transpose  (N_proj, actual_k, W) → (actual_k, N_proj, W)
            # ----------------------------------------------------------
            sinogram_chunk = buffer.transpose(1, 0, 2)   # view, no copy

            # ----------------------------------------------------------
            # 4. write to HDF5
            # ----------------------------------------------------------
            if verbose:
                print(f"  💾 writing rows {h_start}:{h_end} …", end=" ", flush=True)

            t_write = time.time()
            ds[h_start:h_end, :, :] = sinogram_chunk.astype(dtype, copy=False)
            f.flush()

            if verbose:
                print(f"done ({time.time() - t_write:.1f}s)")

            # free buffer explicitly
            del buffer, sinogram_chunk

        # ----------------------------------------------------------------
        # done
        # ----------------------------------------------------------------
        elapsed = time.time() - t_start
        if verbose:
            print(f"\n✅ Finished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
            print(f"   Output file  : {output_path}")
            print(f"   Dataset shape: {ds.shape}")
            print(f"   Dataset dtype: {ds.dtype}")
            actual_mb = output_path.stat().st_size / 1e6
            print(f"   File size    : {fmt_bytes(int(actual_mb * 1e6))}")


if __name__ == "__main__":
    main()
