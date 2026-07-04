from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from pyCGM2.Tools import btkTools


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_points(acq):
    labels = [acq.GetPoint(i).GetLabel() for i in range(acq.GetPointNumber())]
    values = {}
    valid = {}
    residuals = {}
    for label in labels:
        point = acq.GetPoint(label)
        values[label] = point.GetValues().astype(float)
        residuals[label] = point.GetResiduals().copy()
        valid[label] = residuals[label][:, 0] != -1.0
    return labels, values, valid, residuals


def gap_ranges(ok: np.ndarray, start: int, end: int):
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


def run_direction(
    direction: str,
    marker_names,
    unlabeled,
    values,
    valid,
    residuals,
    start_index: int,
    end_index: int,
    first_frame: int,
    max_step_mm: float,
    margin_mm: float,
):
    if direction == "forward":
        frames = range(start_index + 1, end_index + 1)
        prev = lambda frame: frame - 1
        prev2 = lambda frame: frame - 2
    else:
        frames = range(end_index - 1, start_index - 1, -1)
        prev = lambda frame: frame + 1
        prev2 = lambda frame: frame + 2

    rows = []
    skipped = []

    for frame in frames:
        missing_markers = [
            marker for marker in marker_names if not valid[marker][frame] and valid[marker][prev(frame)]
        ]
        grey_points = [raw for raw in unlabeled if valid[raw][frame]]
        if not missing_markers or not grey_points:
            continue

        cost = np.full((len(missing_markers), len(grey_points)), 1e6, dtype=float)
        dist = np.full_like(cost, 1e6)
        for mi, marker in enumerate(missing_markers):
            p = prev(frame)
            p2 = prev2(frame)
            predicted = values[marker][p].copy()
            if 0 <= p2 < len(valid[marker]) and valid[marker][p2]:
                velocity = values[marker][p] - values[marker][p2]
                predicted = values[marker][p] + velocity
                # Fast badminton motion exists, but one-frame marker jumps across body
                # regions are not credible for this labeling pass.
                if np.linalg.norm(velocity) > max_step_mm * 1.5:
                    predicted = values[marker][p]
            for gi, raw in enumerate(grey_points):
                d = float(np.linalg.norm(values[raw][frame] - predicted))
                dist[mi, gi] = d
                if d <= max_step_mm:
                    cost[mi, gi] = d

        row_ind, col_ind = linear_sum_assignment(cost)
        used_markers = set()
        used_raw = set()
        for mi, gi in zip(row_ind, col_ind):
            if cost[mi, gi] >= 1e6:
                continue
            marker = missing_markers[mi]
            raw = grey_points[gi]
            if marker in used_markers or raw in used_raw:
                continue
            marker_sorted = sorted(dist[mi, :])
            raw_sorted = sorted(dist[:, gi])
            marker_margin = marker_sorted[1] - marker_sorted[0] if len(marker_sorted) > 1 else 999.0
            raw_margin = raw_sorted[1] - raw_sorted[0] if len(raw_sorted) > 1 else 999.0
            d = float(dist[mi, gi])
            row = {
                "frame": frame + first_frame,
                "marker": marker,
                "source_unlabeled": raw,
                "distance_mm": round(d, 3),
                "marker_margin_mm": round(float(marker_margin), 3),
                "raw_margin_mm": round(float(raw_margin), 3),
                "direction": direction,
            }
            if marker_margin < margin_mm or raw_margin < margin_mm:
                skipped.append({**row, "reason": "ambiguous_nearest"})
                continue
            values[marker][frame] = values[raw][frame]
            valid[marker][frame] = True
            residuals[marker][frame, 0] = 0.0
            valid[raw][frame] = False
            residuals[raw][frame, 0] = -1.0
            used_markers.add(marker)
            used_raw.add(raw)
            rows.append(row)

    return rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--max-step-mm", type=float, default=90.0)
    parser.add_argument("--margin-mm", type=float, default=20.0)
    parser.add_argument("--write-c3d", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    acq = btkTools.smartReader(str(args.c3d))
    first = acq.GetFirstFrame()
    start_index = args.start_frame - first
    end_index = args.end_frame - first
    labels, values, valid, residuals = load_points(acq)
    marker_names = [label for label in labels if not label.startswith("*")]
    unlabeled = [label for label in labels if label.startswith("*")]
    original_valid = {label: valid[label].copy() for label in marker_names + unlabeled}

    forward_rows, forward_skipped = run_direction(
        "forward",
        marker_names,
        unlabeled,
        values,
        valid,
        residuals,
        start_index,
        end_index,
        first,
        args.max_step_mm,
        args.margin_mm,
    )
    backward_rows, backward_skipped = run_direction(
        "backward",
        marker_names,
        unlabeled,
        values,
        valid,
        residuals,
        start_index,
        end_index,
        first,
        args.max_step_mm,
        args.margin_mm,
    )

    marker_rows = []
    for marker in marker_names:
        before = original_valid[marker][start_index : end_index + 1]
        after = valid[marker][start_index : end_index + 1]
        marker_rows.append(
            {
                "marker": marker,
                "observed_before": int(before.sum()),
                "connected_from_unlabeled": int(after.sum() - before.sum()),
                "missing_after": int((~after).sum()),
                "visible_after": int(after.sum()),
            }
        )

    gap_rows = []
    for marker in marker_names:
        for gs, ge in gap_ranges(valid[marker], start_index, end_index):
            gap_rows.append(
                {
                    "marker": marker,
                    "gap_start_frame": gs + first,
                    "gap_end_frame": ge + first,
                    "gap_length_frames": ge - gs + 1,
                }
            )

    grey_rows = []
    for raw in unlabeled:
        remaining = int(valid[raw][start_index : end_index + 1].sum())
        if remaining:
            grey_rows.append({"unlabeled": raw, "remaining_frames": remaining})

    connected_rows = forward_rows + backward_rows
    skipped_rows = forward_skipped + backward_skipped
    write_csv(args.out_dir / "connected_points.csv", connected_rows)
    write_csv(args.out_dir / "skipped_ambiguous_points.csv", skipped_rows)
    write_csv(args.out_dir / "marker_summary.csv", marker_rows)
    write_csv(args.out_dir / "remaining_marker_gaps.csv", gap_rows)
    write_csv(args.out_dir / "remaining_gray_points.csv", grey_rows)

    output_c3d = ""
    if args.write_c3d:
        output_c3d = str(
            args.out_dir / f"{args.c3d.stem}_bidirectional_connected_{args.start_frame}_{args.end_frame}.c3d"
        )
        for label in marker_names + unlabeled:
            point = acq.GetPoint(label)
            point.SetValues(values[label])
            point.SetResiduals(residuals[label])
        btkTools.smartWriter(acq, output_c3d)

    summary = {
        "input_c3d": str(args.c3d),
        "output_c3d": output_c3d,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "connected_points": len(connected_rows),
        "skipped_ambiguous_points": len(skipped_rows),
        "remaining_marker_gap_segments": len(gap_rows),
        "remaining_gray_frames": int(sum(row["remaining_frames"] for row in grey_rows)),
        "max_step_mm": args.max_step_mm,
        "margin_mm": args.margin_mm,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
