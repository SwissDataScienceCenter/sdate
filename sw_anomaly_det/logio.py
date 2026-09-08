"""Crash-safe, resumable results log.

One row per completed window, appended (never rewritten in place) with a
flush+fsync after every write. No reconstructions are stored, only scalar
scores, so this file stays tiny (KB, not GB) for an entire run -- and because
it's append-only, a RunAI-preemptible job killed mid-run leaves a valid log
with at worst one truncated trailing row, never a corrupted file the way a
single rewritten-in-place checkpoint can be (see ``feedback`` memory from an
earlier incident this project hit with a training checkpoint). On restart,
the trailing row is validated and dropped if torn, and the run resumes right
after the last genuinely complete window -- no windows are silently skipped
or re-run.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional


class AnomalyLog:
    def __init__(self, path, fieldnames: List[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        if self.path.exists() and self.path.stat().st_size > 0:
            _repair_trailing_row(self.path, self.fieldnames)
            existing = _read_header(self.path)
            if existing != self.fieldnames:
                raise ValueError(
                    f"{self.path} has columns {existing}, expected {self.fieldnames} "
                    "-- pass a different --tag/--log_path if the run config changed"
                )
            is_new = False
        else:
            is_new = True
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def last_t(self) -> Optional[int]:
        rows = _valid_rows(self.path, self.fieldnames)
        return int(rows[-1]["t"]) if rows else None

    def append(self, row: Dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def _read_header(path: Path) -> List[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def _row_is_valid(fields: List[str], fieldnames: List[str]) -> bool:
    if len(fields) != len(fieldnames) or any(f == "" for f in fields):
        return False
    try:
        for f in fields:
            float(f)
    except ValueError:
        return False
    return True


def _valid_rows(path: Path, fieldnames: List[str]) -> List[Dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        rows = [dict(zip(fieldnames, r)) for r in reader if _row_is_valid(r, fieldnames)]
    return rows


def _repair_trailing_row(path: Path, fieldnames: List[str]) -> None:
    """Drop a truncated/corrupt trailing row left by a killed job, in place."""
    with open(path, newline="") as fh:
        lines = fh.readlines()
    if len(lines) <= 1:
        return
    last_fields = next(csv.reader([lines[-1]]), [])
    if not _row_is_valid(last_fields, fieldnames):
        with open(path, "w", newline="") as fh:
            fh.writelines(lines[:-1])
