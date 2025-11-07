"""
HEVC Grayscale 10-bit encoding and decoding utilities.

This module provides functions to encode and decode grayscale video tensors
using 10-bit HEVC compression with hardware acceleration when available.
"""

import subprocess
import shutil
from pathlib import Path
import numpy as np
import torch


def encode_hevc_grayscale_10bit(
    v: torch.Tensor,
    outfile: str = "out_grayscale_10bit.mov",
    fps: int = 24,
    cq_hw: int = 90,            # VideoToolbox quality   (0–100, higher = better)
    crf_sw: int = 14,           # libx265 CRF            (0 = lossless)
    preset_sw: str = "slow",# libx265 speed/quality  ("ultrafast" … "placebo")
    tune_grain: bool = False,   # If True, add -tune grain to libx265 (better texture retention)
    force_software: bool = False, # If True, skip hardware attempt and use libx265 directly
    threads: int = 0,            # ffmpeg worker threads (0 lets ffmpeg decide)
    frame_threads: int = None,   # x265 frame-level parallelism (None = default)
    pools: str = None,           # x265 thread pools, e.g. "full" or "8" or "8,4" (None = auto)
    extra_x265: str = ""         # Additional colon-separated x265 params to append
) -> None:
    """
    Encode a grayscale video tensor (T, H, W, float32 0-1) to 10-bit HEVC.

    The function tries Apple VideoToolbox first (≈ real-time) and
    falls back to libx265 if hardware encoding fails.
    
    Args:
        v: Input tensor of shape (T, H, W) with float32 values in range [0, 1]
        outfile: Output file path for the encoded video
        fps: Frames per second for the output video
        cq_hw: VideoToolbox quality (0–100, higher = better quality)
        crf_sw: libx265 CRF value (0 = lossless, higher = more compression)
        preset_sw: libx265 encoding preset for speed/quality tradeoff
        
    Raises:
        ValueError: If input tensor has wrong shape or values out of range
        TypeError: If input tensor is not float32
        RuntimeError: If encoding fails
    """
    if v.ndim != 3:
        raise ValueError("expected a (T, H, W) tensor")

    if v.dtype != torch.float32:
        raise TypeError("input must be torch.float32")
    if torch.isnan(v).any():
        raise ValueError("input contains NaNs")
    if v.min() < 0.0 or v.max() > 1.0:
        raise ValueError("values must be in the range 0–1")

    # ---- 1. 0-1 float → full-range int16 (0-65535) ----
    v16 = (v * 65535.0 + 0.5).clamp_(0, 65535).to(torch.int16).cpu().numpy()

    T, H, W = v16.shape
    print(f"📤  encoding {T} × {W}×{H} frames  (grayscale 10-bit HEVC)")

    if not force_software:
        # ---- 2. command for VideoToolbox (P010 input) ----
        hw_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",          "-pix_fmt", "gray16le",
            "-s", f"{W}x{H}",          "-r", str(fps),
            "-i", "pipe:0",
            "-vf", "format=p010le",
            "-c:v", "hevc_videotoolbox",
            "-profile:v", "main10",
            "-pix_fmt", "p010le",
            "-tag:v", "hvc1",
            "-q:v", str(cq_hw),
            outfile
        ]

        try:
            proc = subprocess.Popen(
                hw_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
            proc.stdin.write(v16.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode()
            ret = proc.wait()

            if ret == 0:
                print("✓ hardware encode succeeded")
                return
            else:
                print("⚠️  hardware path failed, falling back to libx265")
                print(stderr)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg executable not found")
    else:
        print("⏭  force_software=True: skipping hardware encoder")

    # ---- 3. software fallback (libx265) ----
    # Build x265 params string (grayscale keeps mono=1)
    base_params = ["mono=1", "no-info=1", "colorprim=bt709", "transfer=bt709", "colormatrix=bt709"]
    if frame_threads is not None:
        base_params.append(f"frame-threads={frame_threads}")
    if pools is not None:
        base_params.append(f"pools={pools}")
    if extra_x265:
        # Allow user to supply colon-separated list; don't split to preserve ordering
        base_params.append(extra_x265)
    x265_param_str = ":".join(base_params)

    sw_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",          "-pix_fmt", "gray16le",
        "-s", f"{W}x{H}",          "-r", str(fps),
        "-i", "pipe:0",
        "-vf", "format=yuv420p10le",
        "-c:v", "libx265",
        "-profile:v", "main10",
        "-pix_fmt", "yuv420p10le",
        "-preset", preset_sw,
        "-crf", str(crf_sw),
        "-threads", str(threads),
        "-x265-params", x265_param_str,
        outfile
    ]

    if tune_grain:
        # Insert tune option after codec selection for clarity (ffmpeg allows anywhere before output)
        insert_pos = sw_cmd.index("libx265") + 1
        sw_cmd[insert_pos:insert_pos] = ["-tune", "grain"]

    proc = subprocess.Popen(
        sw_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc.stdin.write(v16.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"libx265 failed:\n{stderr}")

    print("✓ software encode (libx265) succeeded")


def decode_hevc_grayscale_10bit(
    infile: str,
    device=None
) -> torch.Tensor:
    """
    Decode a 10-bit HEVC file created by `encode_hevc_grayscale_10bit`
    and return a (T, H, W) float32 tensor in the range 0-1.
    
    Args:
        infile: Path to the input HEVC video file
        device: PyTorch device to place the output tensor on (optional)
        
    Returns:
        torch.Tensor: Decoded video tensor of shape (T, H, W) with float32 values in [0, 1]
        
    Raises:
        RuntimeError: If ffprobe is not found or decoding fails
    """
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe executable not found")

    # ---- 1. query width/height with ffprobe ----
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", infile
    ]
    w_h = subprocess.check_output(probe_cmd).decode().strip()
    if "x" not in w_h:
        raise RuntimeError("unable to read video geometry")
    W, H = map(int, w_h.split("x"))

    # ---- 2. dump raw 16-bit Y plane to stdout ----
    dump_cmd = [
        "ffmpeg", "-v", "error",
        "-i", infile,
        "-pix_fmt", "gray16le",
        "-f", "rawvideo", "pipe:1"
    ]
    raw = subprocess.check_output(dump_cmd)
    frame_bytes = 2 * W * H  # uint16
    if len(raw) % frame_bytes != 0:
        raise RuntimeError("raw dump not an integer number of frames")
    T = len(raw) // frame_bytes

    # ---- 3. uint16 → float32 0-1 ----
    v16 = np.frombuffer(raw, dtype=np.int16).reshape((T, H, W))
    vf = torch.from_numpy(v16).to(torch.float32) / 65535.0
    
    # Move to specified device if provided
    if device is not None:
        vf = vf.to(device)
    
    return vf


def encode_hevc_rgb_10bit(
    v: torch.Tensor,
    outfile: str = "out_rgb_10bit.mov",
    fps: int = 24,
    cq_hw: int = 90,            # VideoToolbox quality (0–100)
    crf_sw: int = 14,           # libx265 CRF (0 = lossless)
    preset_sw: str = "veryslow",# libx265 speed/quality preset
    tune_grain: bool = False,   # If True, add -tune grain in software path
    force_software: bool = False, # If True, skip hardware attempt
    threads: int = 0,            # ffmpeg worker threads (0 = auto)
    frame_threads: int = None,   # x265 frame-level parallelism
    pools: str = None,           # x265 pools setting (e.g. 'full' or '12')
    extra_x265: str = ""         # Additional colon-separated x265 params
) -> None:
    """Encode an RGB video tensor (T, H, W, 3) with float32 0-1 values to 10‑bit HEVC.

    Strategy mirrors `encode_hevc_grayscale_10bit`:
    1. Try hardware (hevc_videotoolbox) by feeding raw rgb48le and converting to p010le.
    2. Fallback to software (libx265) converting to yuv420p10le.

    Args:
        v: Float32 tensor shape (T, H, W, 3), values in [0,1].
        outfile: Output file path.
        fps: Output frame rate.
        cq_hw: Quality for VideoToolbox (lower = smaller file, higher = better).
        crf_sw: CRF for libx265 (0 lossless, 14 visually lossless-ish, higher = more compression).
        preset_sw: libx265 preset (ultrafast … placebo).

    Raises:
        ValueError/TypeError for invalid input.
        RuntimeError if encoding fails.
    """
    if v.ndim != 4 or v.shape[-1] != 3:
        raise ValueError("expected a (T, H, W, 3) tensor")
    if v.dtype != torch.float32:
        raise TypeError("input must be torch.float32")
    if torch.isnan(v).any():
        raise ValueError("input contains NaNs")
    if v.min() < 0.0 or v.max() > 1.0:
        raise ValueError("values must be in the range 0–1")

    # Convert to uint16 full-range per channel (rgb48le)
    v16 = (v * 65535.0 + 0.5).clamp_(0, 65535).to(torch.int16).cpu().numpy()
    T, H, W, C = v16.shape  # C should be 3
    print(f"📤  encoding {T} × {W}×{H} frames  (RGB 10-bit HEVC)")

    # Hardware path: feed rgb48le -> convert to p010le (YUV420 10-bit) inside ffmpeg
    if not force_software:
        hw_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",          "-pix_fmt", "rgb48le",
            "-s", f"{W}x{H}",          "-r", str(fps),
            "-i", "pipe:0",
            "-vf", "format=p010le",
            "-c:v", "hevc_videotoolbox",
            "-profile:v", "main10",
            "-pix_fmt", "p010le",
            "-tag:v", "hvc1",
            "-q:v", str(cq_hw),
            outfile
        ]

        try:
            proc = subprocess.Popen(hw_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.stdin.write(v16.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode()
            ret = proc.wait()
            if ret == 0:
                print("✓ hardware encode (RGB) succeeded")
                return
            else:
                print("⚠️  hardware path failed, falling back to libx265 (RGB)")
                print(stderr)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg executable not found")
    else:
        print("⏭  force_software=True: skipping hardware encoder (RGB)")

    # Software fallback (libx265). We choose yuv420p10le for compatibility.
    # Build x265 params string (RGB)
    base_params = ["no-info=1", "colorprim=bt709", "transfer=bt709", "colormatrix=bt709"]
    if frame_threads is not None:
        base_params.append(f"frame-threads={frame_threads}")
    if pools is not None:
        base_params.append(f"pools={pools}")
    if extra_x265:
        base_params.append(extra_x265)
    x265_param_str = ":".join(base_params)

    sw_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",          "-pix_fmt", "rgb48le",
        "-s", f"{W}x{H}",          "-r", str(fps),
        "-i", "pipe:0",
        "-vf", "format=yuv420p10le",
        "-c:v", "libx265",
        "-profile:v", "main10",
        "-pix_fmt", "yuv420p10le",
        "-preset", preset_sw,
        "-crf", str(crf_sw),
        "-threads", str(threads),
        "-x265-params", x265_param_str,
        outfile
    ]

    if tune_grain:
        insert_pos = sw_cmd.index("libx265") + 1
        sw_cmd[insert_pos:insert_pos] = ["-tune", "grain"]

    proc = subprocess.Popen(sw_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(v16.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"libx265 RGB encode failed:\n{stderr}")
    print("✓ software encode (libx265 RGB) succeeded")


def decode_hevc_rgb_10bit(
    infile: str,
    device=None
) -> torch.Tensor:
    """Decode a 10-bit HEVC file (encoded by `encode_hevc_rgb_10bit`) to (T, H, W, 3) float32 0-1.

    The decoder requests rgb48le (16-bit per channel) from ffmpeg and rescales
    to float in [0,1].

    Args:
        infile: Path to HEVC video.
        device: Optional torch device for output.

    Returns:
        Float32 tensor shape (T, H, W, 3).
    """
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe executable not found")

    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", infile
    ]
    w_h = subprocess.check_output(probe_cmd).decode().strip()
    if "x" not in w_h:
        raise RuntimeError("unable to read video geometry")
    W, H = map(int, w_h.split("x"))

    dump_cmd = [
        "ffmpeg", "-v", "error",
        "-i", infile,
        "-pix_fmt", "rgb48le",
        "-f", "rawvideo", "pipe:1"
    ]
    raw = subprocess.check_output(dump_cmd)
    frame_bytes = 2 * 3 * W * H  # uint16 * RGB
    if len(raw) % frame_bytes != 0:
        raise RuntimeError("raw dump not an integer number of frames")
    T = len(raw) // frame_bytes

    v16 = np.frombuffer(raw, dtype=np.int16).reshape((T, H, W, 3))
    vf = torch.from_numpy(v16).to(torch.float32) / 65535.0
    if device is not None:
        vf = vf.to(device)
    return vf
