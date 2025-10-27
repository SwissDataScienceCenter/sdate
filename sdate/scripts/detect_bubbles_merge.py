#!/usr/bin/env python3
# detect_bubble_merges.py
"""
Detects bubble-merge events in micro-CT movies.

Author: <your-name>
Date:   2025-05-09
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def preprocess(frame_gray: np.ndarray, thresh_val: int = 125) -> np.ndarray:
    """Threshold + morphology → binary mask where bubbles are white."""
    _, mask = cv2.threshold(frame_gray, thresh_val, 255,
                            cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask


def get_components(mask: np.ndarray,
                   min_area: int = 200,
                   max_area: int = 8_000):
    """Yield (label_id, area, centroid_x, centroid_y, contour)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for idx, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        yield idx, area, (cx, cy), c


def match_bubbles(prev, curr, max_dist: float = 40.0):
    """
    Map previous-frame bubbles to current ones by centroid proximity.
    Returns dict {prev_id: curr_id or None}.
    """
    mapping = {}
    for pid, (pa, pxy) in prev.items():
        best = None
        best_dist = max_dist
        for cid, (ca, cxy) in curr.items():
            d = np.hypot(pxy[0] - cxy[0], pxy[1] - cxy[1])
            if d < best_dist:
                best_dist, best = d, cid
        mapping[pid] = best
    return mapping


def detect_merges(prev, curr, mapping,
                  area_factor: float = 0.9):
    """
    Look for ≥2 previous bubbles that now correspond to the *same*
    current bubble and check that the area grew enough.
    """
    # invert mapping: curr_id → [prev_ids...]
    rev = {}
    for pid, cid in mapping.items():
        rev.setdefault(cid, []).append(pid)

    events = []
    for cid, parent_ids in rev.items():
        if cid is None or len(parent_ids) < 2:
            continue
        area_parents = sum(prev[pid][0] for pid in parent_ids)
        area_now = curr[cid][0]
        if area_now >= area_factor * area_parents:
            events.append({
                "frame": None,   # filled in caller
                "curr_id": cid,
                "parent_ids": parent_ids,
                "area_parents": area_parents,
                "area_now": area_now,
                "centroid": curr[cid][1],
            })
    return events


def annotate_frame(frame_bgr, curr_ctr, parent_ctrs):
    """Draw circles around the merged and parent bubbles."""
    for cxy in parent_ctrs:
        cv2.circle(frame_bgr, cxy, 18, (255, 0, 0), 2)   # blue parents
    cv2.circle(frame_bgr, curr_ctr, 22, (0, 0, 255), 3)  # red merged
    return frame_bgr


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    out_dir = Path(args.out_dir)
    if args.annotate:
        out_dir.mkdir(parents=True, exist_ok=True)

    merges = []                 # list of dicts
    prev_bubbles = {}           # {id: (area, (cx,cy))}
    frame_idx = -1
    ring = []                   # circular buffer for ±2 frame context

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = preprocess(gray, thresh_val=args.thresh)

        # collect bubbles in current frame
        bubbles_now = {}
        for bid, area, ctr, _ in get_components(
                mask, args.min_area, args.max_area):
            bubbles_now[bid] = (area, ctr)

        # matching + merge detection (skip very first frame)
        if prev_bubbles:
            mapping = match_bubbles(prev_bubbles, bubbles_now,
                                    max_dist=args.max_centroid_dist)
            events = detect_merges(prev_bubbles, bubbles_now, mapping,
                                   area_factor=args.area_factor)
            for ev in events:
                ev["frame"] = frame_idx
                merges.append(ev)

                # optional annotation
                if args.annotate:
                    # save five-frame clip centred on event
                    clip_frames = ring[-2:] + [frame.copy()]  # prev 2
                    needed = 2                                # next   2
                    # preload next two frames to include in clip
                    for _ in range(needed):
                        ret2, fr = cap.read()
                        if not ret2:
                            break
                        clip_frames.append(fr)
                        ring.append(fr)
                        frame_idx += 1
                    # annotate every frame in clip
                    p_ctrs = [prev_bubbles[pid][1]
                              for pid in ev["parent_ids"]]
                    c_ctr = ev["centroid"]
                    clip_frames = [annotate_frame(f, c_ctr, p_ctrs)
                                   for f in clip_frames]
                    # write clip
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    clip_path = out_dir / f"merge_{ev['frame']:05d}.mp4"
                    h, w = clip_frames[0].shape[:2]
                    vw = cv2.VideoWriter(str(clip_path), fourcc,
                                         args.fps, (w, h))
                    for fr in clip_frames:
                        vw.write(fr)
                    vw.release()

        # housekeeping
        prev_bubbles = bubbles_now
        ring.append(frame)
        ring = ring[-2:]   # keep at most last 2 frames

    cap.release()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if not merges:
        print("No merge events detected.")
        return

    print("\nMerge events:")
    print(" frame |  time (s)  | parents → merged area (px²)")
    print("-------+-----------+-------------------------------")
    for ev in merges:
        t = ev['frame'] / args.fps
        print(f"{ev['frame']:6d} | {t:9.3f} | "
              f"{ev['area_parents']:.0f} → {ev['area_now']:.0f}")

    if args.annotate:
        print(f"\nAnnotated clips written to: {out_dir.resolve()}")


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Detect bubble-merge events in MicroCT movies.")
    p.add_argument("--video", required=True,
                   help="Path to the input movie file (any cv2-readable format).")
    p.add_argument("--fps", type=float, required=True,
                   help="Frames-per-second of the acquisition "
                        "(needed for correct timestamps).")
    p.add_argument("--thresh", type=int, default=125,
                   help="Grayscale threshold for bubble segmentation.")
    p.add_argument("--min_area", type=int, default=200,
                   help="Smallest blob area (px²) to keep.")
    p.add_argument("--max_area", type=int, default=8_000,
                   help="Largest blob area (px²) to keep.")
    p.add_argument("--max_centroid_dist", type=float, default=40.0,
                   help="Max centroid displacement (px) allowed "
                        "between consecutive frames when matching bubbles.")
    p.add_argument("--area_factor", type=float, default=0.90,
                   help="Merged area must be ≥ this × (sum of parent areas).")
    p.add_argument("--annotate", action="store_true",
                   help="Write a short annotated clip (±2 frames) "
                        "around every detected merge.")
    p.add_argument("--out_dir", default="annotated_merges",
                   help="Destination folder for annotated clips.")
    args = p.parse_args()
    main(args)