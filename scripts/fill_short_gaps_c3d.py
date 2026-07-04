from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base


def find_gap_ranges(valid_col: np.ndarray, start_i: int, end_i: int):
    gaps = []
    i = start_i + 1
    while i < end_i:
        if valid_col[i]:
            i += 1
            continue
        gap_start = i
        while i < end_i and not valid_col[i]:
            i += 1
        gaps.append((gap_start, i - 1))
    return gaps


def bracket_gap(valid_col: np.ndarray, gap_start: int, gap_end: int, start_i: int, end_i: int, window: int):
    left = None
    right = None
    for step in range(1, window + 1):
        frame_i = gap_start - step
        if frame_i >= start_i and valid_col[frame_i]:
            left = frame_i
            break
    for step in range(1, window + 1):
        frame_i = gap_end + step
        if frame_i <= end_i and valid_col[frame_i]:
            right = frame_i
            break
    return left, right


def interpolated_support(
    marker: str,
    marker_pos: np.ndarray,
    frame_i: int,
    xyz: np.ndarray,
    valid: np.ndarray,
    label_to_i: dict[str, int],
    refs: dict[tuple[str, str], float],
    neighbors: dict[str, list[str]],
):
    support = 0
    errors = []
    for neighbor in neighbors.get(marker, []):
        neighbor_i = label_to_i[neighbor]
        if not valid[frame_i, neighbor_i]:
            continue
        ref = refs.get((marker, neighbor))
        if ref is None:
            continue
        distance = float(np.linalg.norm(marker_pos - xyz[frame_i, neighbor_i]))
        error = abs(distance - ref)
        tolerance = max(40.0, min(85.0, ref * 0.28))
        if error <= tolerance:
            support += 1
        errors.append(error)
    mean_error = float(np.mean(errors)) if errors else 999.0
    return support, mean_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=2127)
    parser.add_argument("--end-frame", type=int, default=3041)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--search-window", type=int, default=3)
    args = parser.parse_args()

    data, layout, xyz, residual = base.load_c3d(args.c3d)
    marker_names, edges = base.load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    missing = [marker for marker in marker_names if marker not in label_to_i]
    if missing:
        raise ValueError(f"Model markers missing from C3D labels: {missing}")

    valid = base.is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame
    refs = base.build_reference_lengths(xyz, valid, label_to_i, edges, start_i, end_i)
    neighbors = base.neighbor_map(edges)

    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    accepted: list[dict] = []
    rejected: list[dict] = []

    for marker in marker_names:
        marker_i = label_to_i[marker]
        for gap_start, gap_end in find_gap_ranges(valid_out[:, marker_i], start_i, end_i):
            gap_len = gap_end - gap_start + 1
            if gap_len > args.max_gap:
                continue
            left_i, right_i = bracket_gap(
                valid_out[:, marker_i], gap_start, gap_end, start_i, end_i, args.search_window
            )
            if left_i is None or right_i is None:
                rejected.append(
                    {
                        "marker": marker,
                        "gap_start": gap_start + layout.first_frame,
                        "gap_end": gap_end + layout.first_frame,
                        "reason": "no_bracket_within_window",
                    }
                )
                continue

            span = right_i - left_i
            avg_step = float(np.linalg.norm(xyz_out[right_i, marker_i] - xyz_out[left_i, marker_i]) / span)
            if avg_step > base.marker_step_limit(marker) * 0.85:
                rejected.append(
                    {
                        "marker": marker,
                        "gap_start": gap_start + layout.first_frame,
                        "gap_end": gap_end + layout.first_frame,
                        "reason": "motion_too_fast",
                        "avg_step": avg_step,
                    }
                )
                continue

            pending = []
            ok = True
            for frame_i in range(gap_start, gap_end + 1):
                t = (frame_i - left_i) / (right_i - left_i)
                pos = xyz_out[left_i, marker_i] * (1.0 - t) + xyz_out[right_i, marker_i] * t
                support, mean_error = interpolated_support(
                    marker, pos, frame_i, xyz_out, valid_out, label_to_i, refs, neighbors
                )
                if support < base.support_requirement(marker) or mean_error > 55.0:
                    rejected.append(
                        {
                            "marker": marker,
                            "gap_start": gap_start + layout.first_frame,
                            "gap_end": gap_end + layout.first_frame,
                            "frame": frame_i + layout.first_frame,
                            "reason": "weak_model_support",
                            "support": support,
                            "mean_error": mean_error,
                        }
                    )
                    ok = False
                    break
                pending.append((frame_i, pos, support, mean_error))

            if not ok:
                continue

            for frame_i, pos, support, mean_error in pending:
                t = (frame_i - left_i) / (right_i - left_i)
                xyz_out[frame_i, marker_i] = pos
                residual_out[frame_i, marker_i] = (
                    residual_out[left_i, marker_i] * (1.0 - t) + residual_out[right_i, marker_i] * t
                )
                valid_out[frame_i, marker_i] = True
                accepted.append(
                    {
                        "frame": frame_i + layout.first_frame,
                        "marker": marker,
                        "gap_start": gap_start + layout.first_frame,
                        "gap_end": gap_end + layout.first_frame,
                        "left_frame": left_i + layout.first_frame,
                        "right_frame": right_i + layout.first_frame,
                        "gap_len": gap_len,
                        "avg_step": avg_step,
                        "support": support,
                        "mean_error": mean_error,
                        "method": "short_gap_linear_interpolation",
                    }
                )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "accepted_short_gap_fill.csv", accepted)
    base.write_csv(args.report_dir / "rejected_short_gap_fill.csv", rejected)

    after_valid = base.is_valid(xyz_out, residual_out)
    summary = []
    for marker in marker_names:
        marker_i = label_to_i[marker]
        before_missing = int(np.sum(~valid[start_i : end_i + 1, marker_i]))
        after_missing = int(np.sum(~after_valid[start_i : end_i + 1, marker_i]))
        if before_missing != after_missing:
            summary.append(
                {
                    "marker": marker,
                    "before_missing": before_missing,
                    "after_missing": after_missing,
                    "filled": before_missing - after_missing,
                }
            )
    base.write_csv(args.report_dir / "missing_summary_after_short_gap_fill.csv", summary)

    print(f"accepted={len(accepted)}")
    print(f"rejected={len(rejected)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
