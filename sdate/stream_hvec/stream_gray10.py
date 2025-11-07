"""
Streaming HEVC (H.265) encoding utilities for grayscale 10-bit content.

Provides a file-segment oriented streaming API that can encode extremely
large sequences without keeping all frames in memory.

- HevcGray10Streamer: incremental appends using ffmpeg stdin piping
  with automatic hardware (VideoToolbox) attempt and libx265 fallback.
- concat_hevc_segments: lossless container-level concatenation (-c copy)
  of segments sharing codec/parameters; optional reencode fallback.

This mirrors the single-shot functions in `sdate.video_compression.hevc_grayscale`
while offering a stateful, streaming interface.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Iterable

import numpy as np
import torch


@dataclass
class EncoderParams:
    fps: int = 24
    cq_hw: int = 90
    crf_sw: int = 14
    preset_sw: str = "veryslow"
    tune_grain: bool = False
    force_software: bool = False
    threads: int = 0
    frame_threads: Optional[int] = None
    pools: Optional[str] = None
    extra_x265: str = ""


class HevcGray10Streamer:
    """Stream grayscale frames to HEVC 10-bit segments.

    Usage:
        streamer = HevcGray10Streamer(base_path="/tmp/out", segment_prefix="clip")
        with streamer.start_segment(q=90):  # start first file
            for frame in frames:  # frame: torch.float32 (H,W) in [0,1]
                streamer.append_frame(frame)
        # start a new segment with different quality
        with streamer.start_segment(q=75):
            ...
        # finally concat
        concat_hevc_segments(streamer.segments, "/tmp/out_all.mov")

    Notes:
    - Hardware path uses hevc_videotoolbox with p010le conversion.
    - Software path uses libx265 main10 yuv420p10le.
    - Input frames must be torch.float32, range [0,1], shape (H,W).
    - Each segment can have different quality; all will be concatenated later.
    """

    def __init__(
        self,
        base_path: os.PathLike | str,
        segment_prefix: str = "segment",
        params: Optional[EncoderParams] = None,
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.segment_prefix = segment_prefix
        self.params = params or EncoderParams()
        self._proc: Optional[subprocess.Popen] = None
        self._current_outfile: Optional[Path] = None
        self._hw_attempted = False
        self._using_hardware = False
        self._shape: Optional[tuple[int, int]] = None  # (H, W)
        self._frame_count = 0
        self.segments: List[Path] = []
        self._segment_open = False

    # ------------- segment lifecycle -------------
    def start_segment(self, q: Optional[int] = None, outfile: Optional[str | os.PathLike] = None):
        """Start a new output file (segment).

        - q: sets VideoToolbox quality if hardware path is used; otherwise
             software CRF is mapped approximately via crf_sw when q is None.
             If provided, this call updates the params for this segment only.
        - outfile: optional custom file name; default is f"{prefix}_{N:04d}.mov".

        Returns a context manager; on exit it automatically closes the segment.
        """
        if self._proc is not None:
            raise RuntimeError("segment already open; close it before starting a new one")

        seg_index = len(self.segments)
        if outfile is None:
            outfile_path = self.base_path / f"{self.segment_prefix}_{seg_index:04d}.mov"
        else:
            outfile_path = self.base_path / Path(outfile)
        outfile_path.parent.mkdir(parents=True, exist_ok=True)

        p = EncoderParams(**asdict(self.params))
        if q is not None:
            p.cq_hw = int(q)
            # heuristic mapping: when forcing software later, keep existing crf_sw
        self._current_outfile = outfile_path
        self._frame_count = 0
        self._shape = None
        self._prepare_ffmpeg(outfile_path, p)
        self._segment_open = True

        class _Ctx:
            def __init__(self, outer: "HevcGray10Streamer"):
                self.outer = outer
            def __enter__(self):
                return self.outer
            def __exit__(self, exc_type, exc, tb):
                self.outer.close_segment()
        return _Ctx(self)

    def _prepare_ffmpeg(self, outfile: Path, p: EncoderParams) -> None:
        # decide initial attempt path (hardware unless forced)
        self._hw_attempted = not p.force_software
        self._using_hardware = False

        # Build commands for both paths; we'll try hardware first if allowed
        hw_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray16le",
            "-s", "{W}x{H}", "-r", str(p.fps),
            "-i", "pipe:0",
            "-vf", "format=p010le",
            "-c:v", "hevc_videotoolbox",
            "-profile:v", "main10",
            "-pix_fmt", "p010le",
            "-tag:v", "hvc1",
            "-q:v", str(p.cq_hw),
            str(outfile)
        ]
        # placeholders {W}x{H} are replaced on first frame when known

        # x265 parameters
        base_params = ["mono=1", "no-info=1", "colorprim=bt709", "transfer=bt709", "colormatrix=bt709"]
        if p.frame_threads is not None:
            base_params.append(f"frame-threads={p.frame_threads}")
        if p.pools is not None:
            base_params.append(f"pools={p.pools}")
        if p.extra_x265:
            base_params.append(p.extra_x265)
        x265_param_str = ":".join(base_params)

        sw_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray16le",
            "-s", "{W}x{H}", "-r", str(p.fps),
            "-i", "pipe:0",
            "-vf", "format=yuv420p10le",
            "-c:v", "libx265",
            "-profile:v", "main10",
            "-tag:v", "hvc1",
            "-pix_fmt", "yuv420p10le",
            "-preset", p.preset_sw,
            "-crf", str(p.crf_sw),
            "-threads", str(p.threads),
            "-x265-params", x265_param_str,
            str(outfile)
        ]

        if p.tune_grain:
            insert_pos = sw_cmd.index("libx265") + 1
            sw_cmd[insert_pos:insert_pos] = ["-tune", "grain"]

        # save both command variants; we can't start the process until we know WxH
        self._pending_hw_cmd = hw_cmd
        self._pending_sw_cmd = sw_cmd
        self._pending_params = p
        self._proc = None

    # ------------- append -------------
    def append_frame(self, frame: torch.Tensor) -> None:
        """Append a single grayscale frame (H, W), float32 in [0,1].

        On the first frame of each segment we finalize geometry and spawn
        the ffmpeg process. Subsequent frames are written to stdin.
        """
        if frame.ndim != 2:
            raise ValueError("expected frame shape (H, W)")
        if frame.dtype != torch.float32:
            raise TypeError("frame must be torch.float32")
        if torch.isnan(frame).any():
            raise ValueError("frame contains NaNs")
        if frame.min() < 0.0 or frame.max() > 1.0:
            raise ValueError("frame values must be in [0,1]")

        # convert to uint16 little-endian buffer
        f16 = (frame * 65535.0 + 0.5).clamp_(0, 65535).to(torch.int16).cpu().numpy()
        H, W = f16.shape

        if not self._segment_open or self._current_outfile is None:
            raise RuntimeError("no open segment; call start_segment() first")

        if self._shape is None:
            self._shape = (H, W)
            # spawn appropriate ffmpeg with known geometry
            self._proc = self._start_ffmpeg_with_geometry(W, H)

        else:
            if self._shape != (H, W):
                raise ValueError(f"frame size {H}x{W} does not match open segment {self._shape}")

        try:
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(f16.tobytes())
            self._frame_count += 1
        except BrokenPipeError as e:
            # If the hardware path breaks on the very first frame, fallback to software now
            if self._using_hardware and self._frame_count == 0:
                try:
                    if self._proc and self._proc.stderr:
                        _ = self._proc.stderr.read()
                except Exception:
                    pass
                if self._proc:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                # switch to software and retry writing this first frame
                self._hw_attempted = False
                self._using_hardware = False
                self._proc = self._start_ffmpeg_with_geometry(W, H)
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(f16.tobytes())
                self._frame_count += 1
                return
            stderr = ""
            if self._proc and self._proc.stderr:
                try:
                    stderr = self._proc.stderr.read().decode(errors="ignore")
                except Exception:
                    pass
            raise RuntimeError(f"ffmpeg pipe closed unexpectedly after {self._frame_count} frames.\n{stderr}") from e

    def _start_ffmpeg_with_geometry(self, W: int, H: int) -> subprocess.Popen:
        p = self._pending_params
        assert p is not None
        # materialize command by replacing geometry placeholders
        def materialize(cmd: List[str]) -> List[str]:
            out = []
            for tok in cmd:
                if tok == "{W}x{H}":
                    out.append(f"{W}x{H}")
                else:
                    out.append(tok)
            return out

        proc = None
        if self._hw_attempted:
            cmd = materialize(self._pending_hw_cmd)
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self._using_hardware = True
            except FileNotFoundError:
                raise RuntimeError("ffmpeg executable not found")
        else:
            # directly start software
            cmd = materialize(self._pending_sw_cmd)
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self._using_hardware = False
            except FileNotFoundError:
                raise RuntimeError("ffmpeg executable not found")
        return proc

    # ------------- rotate / close -------------
    def close_segment(self) -> None:
        if not self._segment_open:
            return
        if self._proc is None:
            # no frames were written; simply clear state without producing a file
            self._current_outfile = None
            self._shape = None
            self._frame_count = 0
            self._segment_open = False
            return
        # flush and finalize
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        stderr = b""
        if self._proc.stderr:
            try:
                stderr = self._proc.stderr.read()
            except Exception:
                pass
        ret = self._proc.wait()
        using_hw = self._using_hardware
        outfile = self._current_outfile

        # clear state (will mark as closed below)
        self._proc = None
        self._shape = None
        self._using_hardware = False
        self._hw_attempted = False

        if ret != 0:
            # If hardware path failed on finalize, try software fallback
            if using_hw:
                # retry this segment with software; we need to delete partially written file and restart
                if outfile and outfile.exists():
                    try:
                        outfile.unlink()
                    except Exception:
                        pass
                # We cannot resend past frames; in streaming scenarios, the safe behavior
                # is to surface an error immediately so the caller can handle retry logic.
                raise RuntimeError("hardware encoder failed during finalize; cannot recover in streaming mode. Please set force_software=True for this segment and resend frames.")
            else:
                raise RuntimeError(f"libx265 failed during finalize. stderr:\n{stderr.decode(errors='ignore')}")

        if outfile is not None:
            self.segments.append(outfile)
        self._current_outfile = None
        self._segment_open = False

    def rotate_segment(self, q: Optional[int] = None, outfile: Optional[str | os.PathLike] = None) -> None:
        """Close current segment (if any) and start a new one with optional quality change."""
        self.close_segment()
        self.start_segment(q=q, outfile=outfile)

    # ------------- utilities -------------
    def current_outfile(self) -> Optional[Path]:
        return self._current_outfile



    


# --------- Functional wrappers for a function-style API ---------
def open_hevc_gray10_stream(
    base_path: os.PathLike | str,
    segment_prefix: str = "segment",
    params: Optional[EncoderParams] = None,
    q: Optional[int] = None,
    outfile: Optional[str | os.PathLike] = None,
) -> HevcGray10Streamer:
    streamer = HevcGray10Streamer(base_path=base_path, segment_prefix=segment_prefix, params=params)
    # start first segment immediately; caller can ignore the returned context manager
    streamer.start_segment(q=q, outfile=outfile)
    return streamer


def append_hevc_gray10_frame(streamer: HevcGray10Streamer, frame: torch.Tensor) -> None:
    streamer.append_frame(frame)


def rotate_hevc_gray10_stream(
    streamer: HevcGray10Streamer,
    q: Optional[int] = None,
    outfile: Optional[str | os.PathLike] = None,
) -> None:
    streamer.rotate_segment(q=q, outfile=outfile)


def close_hevc_gray10_stream(streamer: HevcGray10Streamer) -> None:
    streamer.close_segment()


def concat_hevc_segments(
    files: Iterable[os.PathLike | str],
    outfile: os.PathLike | str
) -> None:
    """Concatenate a list of HEVC files into one output.

    Performs container-level concatenation using ffmpeg's concat demuxer with
    codec copy. All input segments must share identical codec parameters
    (codec, profile, resolution, pixel format, time base) for a seamless copy.

    Args:
        files: Iterable of input file paths (e.g., .mov or .mp4 with HEVC video).
        outfile: Destination output file path.

    Raises:
        FileNotFoundError: If any input path does not exist.
        ValueError: If the list is empty.
        RuntimeError: If ffmpeg fails to concatenate.
    """
    paths = [Path(f).resolve() for f in files]
    if not paths:
        raise ValueError("no input files provided to concat_hevc_segments")
    for p in paths:
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"input segment not found: {p}")

    out_path = Path(outfile).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create concat list file
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        list_file = Path(tf.name)
        for p in paths:
            s = str(p)
            s = s.replace("'", "'\\''")
            tf.write(f"file '{s}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-tag:v", "hvc1",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="ignore")
            raise RuntimeError(f"ffmpeg concat failed (copy).\n{err}")
    finally:
        try:
            list_file.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
