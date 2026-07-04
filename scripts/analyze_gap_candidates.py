from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyCGM2.Tools import btkTools


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gap_ranges(ok, start, end):
    gaps = []
    i = start
    while i <= end:
        if ok[i]:
            i += 1
            continue
        s = i
        while i <= end and not ok[i]:
            i += 1
        gaps.append((s, i - 1))
    return gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    args = parser.parse_args()

    acq = btkTools.smartReader(str(args.c3d))
    first = acq.GetFirstFrame()
    start = args.start_frame - first
    end = args.end_frame - first
    labels = [acq.GetPoint(i).GetLabel() for i in range(acq.GetPointNumber())]
    named = [label for label in labels if not label.startswith("*")]
    unlabeled = [label for label in labels if label.startswith("*")]
    values = {}
    valid = {}
    for label in labels:
        point = acq.GetPoint(label)
        values[label] = point.GetValues().astype(float)
        valid[label] = point.GetResiduals()[:, 0] != -1.0

    rows = []
    for marker in named:
        for gs, ge in gap_ranges(valid[marker], start, end):
            if ge - gs + 1 < 3:
                continue
            candidates = []
            left_frame = gs - 1 if gs - 1 >= 0 and valid[marker][gs - 1] else None
            right_frame = ge + 1 if ge + 1 < acq.GetPointFrameNumber() and valid[marker][ge + 1] else None
            for raw in unlabeled:
                frames = np.where(valid[raw][gs : ge + 1])[0] + gs
                if frames.size == 0:
                    continue
                first_raw = int(frames[0])
                last_raw = int(frames[-1])
                coverage = int(frames.size)
                left_distance = ""
                right_distance = ""
                if left_frame is not None:
                    left_distance = float(
                        np.linalg.norm(values[raw][first_raw] - values[marker][left_frame])
                    )
                if right_frame is not None:
                    right_distance = float(
                        np.linalg.norm(values[raw][last_raw] - values[marker][right_frame])
                    )
                distances = [d for d in [left_distance, right_distance] if d != ""]
                score = (
                    sum(distances) / len(distances) if distances else 999999.0
                ) - coverage * 0.05
                candidates.append(
                    {
                        "marker": marker,
                        "gap_start_frame": gs + first,
                        "gap_end_frame": ge + first,
                        "gap_length": ge - gs + 1,
                        "unlabeled": raw,
                        "coverage": coverage,
                        "raw_first_frame": first_raw + first,
                        "raw_last_frame": last_raw + first,
                        "left_distance_mm": round(left_distance, 3)
                        if left_distance != ""
                        else "",
                        "right_distance_mm": round(right_distance, 3)
                        if right_distance != ""
                        else "",
                        "score": round(score, 3),
                    }
                )
            candidates.sort(key=lambda row: row["score"])
            rows.extend(candidates[:5])

    write_csv(args.out, rows)
    print(args.out)


if __name__ == "__main__":
    main()
