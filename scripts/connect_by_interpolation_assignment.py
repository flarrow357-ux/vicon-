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


def build_predictions(marker_names, values, original_valid, start_index, end_index):
    predictions = {marker: values[marker].copy() for marker in marker_names}
    predictable = {marker: np.zeros(values[marker].shape[0], dtype=bool) for marker in marker_names}
    for marker in marker_names:
        ok = original_valid[marker]
        for gs, ge in gap_ranges(ok, start_index, end_index):
            left = gs - 1
            right = ge + 1
            if left < 0 or right >= len(ok) or not ok[left] or not ok[right]:
                continue
            for frame in range(gs, ge + 1):
                alpha = (frame - left) / (right - left)
                predictions[marker][frame] = (1.0 - alpha) * values[marker][left] + alpha * values[marker][right]
                predictable[marker][frame] = True
    return predictions, predictable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--distance-threshold-mm", type=float, default=220.0)
    parser.add_argument("--review-threshold-mm", type=float, default=120.0)
    parser.add_argument("--margin-threshold-mm", type=float, default=25.0)
    parser.add_argument("--allow-review", action="store_true")
    parser.add_argument("--write-c3d", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    acq = btkTools.smartReader(str(args.c3d))
    first = acq.GetFirstFrame()
    start_index = args.start_frame - first
    end_index = args.end_frame - first
    if start_index < 0 or end_index >= acq.GetPointFrameNumber():
        raise ValueError("帧范围超出 C3D")

    labels, values, valid, residuals = load_points(acq)
    marker_names = [label for label in labels if not label.startswith("*")]
    unlabeled = [label for label in labels if label.startswith("*")]
    original_valid = {label: valid[label].copy() for label in marker_names + unlabeled}
    predictions, predictable = build_predictions(
        marker_names, values, original_valid, start_index, end_index
    )

    connected_rows = []
    review_rows = []

    for frame in range(start_index, end_index + 1):
        missing_markers = [
            marker
            for marker in marker_names
            if not valid[marker][frame] and predictable[marker][frame]
        ]
        grey_points = [raw for raw in unlabeled if valid[raw][frame]]
        if not missing_markers or not grey_points:
            continue

        cost = np.full((len(missing_markers), len(grey_points)), 1e6, dtype=float)
        raw_dist = np.zeros_like(cost)
        for mi, marker in enumerate(missing_markers):
            predicted = predictions[marker][frame]
            for gi, raw in enumerate(grey_points):
                d = float(np.linalg.norm(values[raw][frame] - predicted))
                raw_dist[mi, gi] = d
                if d <= args.distance_threshold_mm:
                    cost[mi, gi] = d

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_markers = set()
        assigned_raw = set()
        for mi, gi in zip(row_ind, col_ind):
            if cost[mi, gi] >= 1e6:
                continue
            marker = missing_markers[mi]
            raw = grey_points[gi]
            distance = float(raw_dist[mi, gi])

            marker_sorted = sorted(raw_dist[mi, :])
            raw_sorted = sorted(raw_dist[:, gi])
            marker_margin = (
                float(marker_sorted[1] - marker_sorted[0])
                if len(marker_sorted) > 1
                else 999.0
            )
            raw_margin = (
                float(raw_sorted[1] - raw_sorted[0])
                if len(raw_sorted) > 1
                else 999.0
            )
            needs_review = (
                distance > args.review_threshold_mm
                or marker_margin < args.margin_threshold_mm
                or raw_margin < args.margin_threshold_mm
            )

            if marker in assigned_markers or raw in assigned_raw:
                continue
            row = {
                "frame": frame + first,
                "marker": marker,
                "source_unlabeled": raw,
                "distance_mm": round(distance, 3),
                "marker_margin_mm": round(marker_margin, 3),
                "raw_margin_mm": round(raw_margin, 3),
                "needs_review": needs_review,
            }
            if needs_review:
                review_rows.append(row)
                if not args.allow_review:
                    continue
            values[marker][frame] = values[raw][frame]
            valid[marker][frame] = True
            residuals[marker][frame, 0] = 0.0
            valid[raw][frame] = False
            residuals[raw][frame, 0] = -1.0
            assigned_markers.add(marker)
            assigned_raw.add(raw)
            connected_rows.append(row)

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

    write_csv(args.out_dir / "assigned_gray_points.csv", connected_rows)
    write_csv(args.out_dir / "needs_review.csv", review_rows)
    write_csv(args.out_dir / "marker_summary.csv", marker_rows)
    write_csv(args.out_dir / "remaining_marker_gaps.csv", gap_rows)
    write_csv(args.out_dir / "remaining_gray_points.csv", grey_rows)

    output_c3d = ""
    if args.write_c3d:
        output_c3d = str(
            args.out_dir
            / f"{args.c3d.stem}_interp_connected_{args.start_frame}_{args.end_frame}.c3d"
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
        "review_points": len(review_rows),
        "remaining_marker_gap_segments": len(gap_rows),
        "remaining_gray_frames": int(sum(row["remaining_frames"] for row in grey_rows)),
        "distance_threshold_mm": args.distance_threshold_mm,
        "review_threshold_mm": args.review_threshold_mm,
        "margin_threshold_mm": args.margin_threshold_mm,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
