from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base


def segment_rows(rows: list[dict]):
    rows = sorted(rows, key=lambda row: int(row["frame"]))
    segment = []
    previous = None
    for row in rows:
        frame = int(row["frame"])
        if previous is not None and frame != previous + 1:
            yield segment
            segment = []
        segment.append(row)
        previous = frame
    if segment:
        yield segment


def read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--held", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    args = parser.parse_args()

    data, layout, xyz, residual = base.load_c3d(args.c3d)
    markers, edges = base.load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    valid = base.is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame
    refs = base.build_reference_lengths(xyz, valid, label_to_i, edges, start_i, end_i)
    neighbors = base.neighbor_map(edges)

    grouped = defaultdict(list)
    for row in read_csv(args.held):
        grouped[(row["marker"], row["raw"])].append(row)

    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    accepted = []
    rejected = []

    for (marker, raw), rows in grouped.items():
        marker_i = label_to_i[marker]
        raw_i = label_to_i[raw]
        for segment in segment_rows(rows):
            frames = [int(row["frame"]) for row in segment]
            frame_i = [frame - layout.first_frame for frame in frames]
            points = xyz[frame_i, raw_i]
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.array([])

            left_i = None
            right_i = None
            for probe in range(frame_i[0] - 1, start_i - 1, -1):
                if valid[probe, marker_i]:
                    left_i = probe
                    break
            for probe in range(frame_i[-1] + 1, end_i + 1):
                if valid[probe, marker_i]:
                    right_i = probe
                    break

            left_jump = float(np.linalg.norm(points[0] - xyz[left_i, marker_i])) if left_i is not None else 9999.0
            right_jump = float(np.linalg.norm(points[-1] - xyz[right_i, marker_i])) if right_i is not None else 9999.0
            max_step = float(np.max(steps)) if steps.size else 0.0
            max_pred = max(float(row["pred_dist"]) for row in segment)
            max_report_segerr = max(float(row["seg_error"]) for row in segment)

            supports = []
            model_errors = []
            for fi, point in zip(frame_i, points):
                support = 0
                local_errors = []
                for neighbor in neighbors.get(marker, []):
                    neighbor_i = label_to_i[neighbor]
                    if not valid[fi, neighbor_i]:
                        continue
                    ref = refs.get((marker, neighbor))
                    if ref is None:
                        continue
                    error = abs(float(np.linalg.norm(point - xyz[fi, neighbor_i])) - ref)
                    local_errors.append(error)
                    if error <= max(40.0, min(85.0, ref * 0.28)):
                        support += 1
                supports.append(support)
                if local_errors:
                    model_errors.append(float(np.mean(local_errors)))
            min_support = min(supports) if supports else 0
            max_model_error = max(model_errors) if model_errors else 9999.0

            metrics = {
                "marker": marker,
                "raw": raw,
                "start": frames[0],
                "end": frames[-1],
                "segment_len": len(frames),
                "left_frame": left_i + layout.first_frame if left_i is not None else "",
                "right_frame": right_i + layout.first_frame if right_i is not None else "",
                "left_jump": left_jump,
                "right_jump": right_jump,
                "max_step": max_step,
                "max_pred": max_pred,
                "max_report_segerr": max_report_segerr,
                "max_model_err": max_model_error,
                "min_support": min_support,
            }
            ok = (
                min_support >= 2
                and max_report_segerr <= 24.0
                and max_model_error <= 24.0
                and max_step <= 45.0
                and left_jump <= 80.0
                and right_jump <= 80.0
                and max_pred <= 55.0
            )
            if not ok:
                rejected.append({**metrics, "reason": "failed_long_segment_audit"})
                continue

            for row in segment:
                fi = int(row["frame"]) - layout.first_frame
                if valid_out[fi, marker_i] or not valid_out[fi, raw_i]:
                    rejected.append({**metrics, "frame": row["frame"], "reason": "point_state_changed"})
                    continue
                xyz_out[fi, marker_i] = xyz_out[fi, raw_i]
                residual_out[fi, marker_i] = residual_out[fi, raw_i]
                xyz_out[fi, raw_i] = 0.0
                residual_out[fi, raw_i] = -1.0
                valid_out[fi, marker_i] = True
                valid_out[fi, raw_i] = False
                accepted.append({**row, **metrics, "method": "checked_long_segment_connect"})

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "accepted_checked_long_segments.csv", accepted)
    base.write_csv(args.report_dir / "rejected_checked_long_segments.csv", rejected)

    after_valid = base.is_valid(xyz_out, residual_out)
    summary = []
    for marker in markers:
        marker_i = label_to_i[marker]
        before_missing = int(np.sum(~valid[start_i : end_i + 1, marker_i]))
        after_missing = int(np.sum(~after_valid[start_i : end_i + 1, marker_i]))
        if before_missing != after_missing:
            summary.append(
                {
                    "marker": marker,
                    "before_missing": before_missing,
                    "after_missing": after_missing,
                    "connected": before_missing - after_missing,
                }
            )
    base.write_csv(args.report_dir / "missing_summary_after_checked_long_segments.csv", summary)
    print(f"accepted={len(accepted)}")
    print(f"rejected_segments_or_rows={len(rejected)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
