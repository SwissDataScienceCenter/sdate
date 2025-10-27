import os
from pathlib import Path
import torch

from sdate.stream_hvec import HevcGray10Streamer, concat_hevc_segments


def main():
    outdir = Path("./scripts/compression_sweep_results/stream_demo").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    streamer = HevcGray10Streamer(outdir, segment_prefix="demo")
    # segment 1 at q=90
    with streamer.start_segment(q=90):
        for i in range(30):
            # synthetic gradient frame
            H, W = 256, 256
            y = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
            x = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
            frame = (0.5 * x + 0.5 * y).to(torch.float32).clamp(0, 1)
            streamer.append_frame(frame)
    # segment 2 at q=70
    with streamer.start_segment(q=70):
        for i in range(30):
            H, W = 256, 256
            frame = torch.rand(H, W, dtype=torch.float32)
            streamer.append_frame(frame)

    all_out = outdir / "demo_all.mov"
    concat_hevc_segments(streamer.segments, all_out)
    print(f"Wrote {len(streamer.segments)} segments -> {all_out}")


if __name__ == "__main__":
    main()
