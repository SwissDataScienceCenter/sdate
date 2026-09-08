"""Cross-angle aggregation: majority voting (and a hook for 3D triangulation).

Each tracked angle produces its own anomaly timeline indexed by *turn* (one turn
= one full rotation ≈ the same wall-clock instant across angles).  An anomaly
seen from a single viewpoint may be a projection artefact; one seen from several
angles at the same turn is far more likely real.  This module aligns the
per-angle timelines on the common turn range and counts votes.

Triangulation (localising the anomaly in 3D from the per-angle mask centroids +
known angles) is left as a documented extension point — the per-mask centroids
and the calibrated angles are exactly the inputs a back-projection needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class VoteResult:
    turns: np.ndarray          # common turn indices
    t2_votes: np.ndarray       # # angles flagging T² at each turn
    q_votes: np.ndarray        # # angles flagging Q at each turn
    n_angles: int
    t2_confirmed: np.ndarray   # bool: votes >= min_votes
    q_confirmed: np.ndarray
    min_votes: int


def aggregate_votes(results: Dict[int, "AngleResults"], min_votes: Optional[int] = None) -> VoteResult:
    """Majority-vote anomaly flags across angles, aligned by turn index."""
    per_angle = {ai: {int(s.turn): s for s in r.scores} for ai, r in results.items()}
    all_turns = sorted(set().union(*[set(d) for d in per_angle.values()])) if per_angle else []
    turns = np.array(all_turns, dtype=int)
    n_angles = len(results)
    if min_votes is None:
        min_votes = n_angles // 2 + 1          # strict majority

    t2v = np.zeros(len(turns), dtype=int)
    qv = np.zeros(len(turns), dtype=int)
    for ti, t in enumerate(turns):
        for d in per_angle.values():
            s = d.get(int(t))
            if s is None or not s.scored:
                continue
            t2v[ti] += int(s.t2_flag)
            qv[ti] += int(s.q_flag)

    return VoteResult(
        turns=turns, t2_votes=t2v, q_votes=qv, n_angles=n_angles,
        t2_confirmed=t2v >= min_votes, q_confirmed=qv >= min_votes,
        min_votes=min_votes,
    )


def mask_centroid(mask: np.ndarray, frac: float = 0.9) -> tuple:
    """Intensity-weighted centroid ``(row, col)`` of a contribution mask.

    Thresholds at the ``frac`` quantile first so the centroid tracks the hot
    region, not the diffuse background.  This is the per-view input a future
    triangulation step back-projects through the known angle to localise the
    anomaly in 3D.
    """
    m = np.abs(np.asarray(mask, dtype=np.float64))
    if m.max() <= 0:
        return (np.nan, np.nan)
    thr = np.quantile(m, frac)
    w = np.where(m >= thr, m, 0.0)
    if w.sum() <= 0:
        return (np.nan, np.nan)
    rr, cc = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    return (float((w * rr).sum() / w.sum()), float((w * cc).sum() / w.sum()))
