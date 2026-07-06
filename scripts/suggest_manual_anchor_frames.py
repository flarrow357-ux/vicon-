from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base


def missing_run_length(valid: np.ndarray, frame_i: int, marker_i: int, start_i: int, end_i: int) -> int:
    left = frame_i
    while left - 1 >= start_i and not valid[left - 1, marker_i]:
        left -= 1
    right = frame_i
    while right + 1 <= end_i and not valid[right + 1, marker_i]:
        right += 1
    return right - left + 1


def choose_spaced(rows: list[dict], top_n: int, min_frame_gap: int) -> list[dict]:
    chosen: list[dict] = []
    for row in rows:
        frame = int(row["frame"])
        if any(abs(frame - int(prev["frame"])) < min_frame_gap for prev in chosen):
            continue
        chosen.append({**row, "suggest_rank": len(chosen) + 1})
        if len(chosen) >= top_n:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-frame-gap", type=int, default=20)
    args = parser.parse_args()

    _data, layout, xyz, residual = base.load_c3d(args.c3d)
    marker_names, _edges = base.load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    missing_model_markers = [marker for marker in marker_names if marker not in label_to_i]
    if missing_model_markers:
        raise ValueError(f"Model markers missing from C3D labels: {missing_model_markers}")

    valid = base.is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame
    if start_i < 0 or end_i >= valid.shape[0] or start_i > end_i:
        raise ValueError("Invalid start/end frame for this C3D.")

    raw_indices = [i for i, label in enumerate(layout.labels) if label.startswith("*")]
    marker_indices = [label_to_i[marker] for marker in marker_names]

    missing_counts = np.zeros(valid.shape[0], dtype=int)
    raw_counts = np.zeros(valid.shape[0], dtype=int)
    for frame_i in range(start_i, end_i + 1):
        missing_counts[frame_i] = int(np.sum(~valid[frame_i, marker_indices]))
        raw_counts[frame_i] = int(np.sum(valid[frame_i, raw_indices]))

    rows: list[dict] = []
    for frame_i in range(start_i + 1, end_i):
        missing_markers = [marker for marker in marker_names if not valid[frame_i, label_to_i[marker]]]
        missing_count = len(missing_markers)
        raw_count = int(raw_counts[frame_i])
        convertible_upper_bound = min(missing_count, raw_count)
        if missing_count == 0 or raw_count == 0:
            continue

        win_start = max(start_i, frame_i - args.window)
        win_end = min(end_i, frame_i + args.window)
        neighbor_missing = int(np.sum(missing_counts[win_start : win_end + 1]) - missing_count)
        neighbor_raw = int(np.sum(raw_counts[win_start : win_end + 1]) - raw_count)
        run_total = int(
            sum(missing_run_length(valid, frame_i, label_to_i[marker], start_i, end_i) for marker in missing_markers)
        )

        # A complete manual frame is most useful when many labels are missing,
        # enough grey points exist in the same frame, and the surrounding frames
        # also have missing labels that can propagate from this new anchor.
        score = float(convertible_upper_bound * 1000 + run_total * 15 + neighbor_missing * 3 + raw_count)
        rows.append(
            {
                "frame": frame_i + layout.first_frame,
                "score": round(score, 3),
                "missing_human_count": missing_count,
                "grey_point_count": raw_count,
                "convertible_upper_bound": convertible_upper_bound,
                "neighbor_missing_human_count": neighbor_missing,
                "neighbor_grey_point_count": neighbor_raw,
                "missing_run_total": run_total,
                "window": args.window,
                "missing_markers": ";".join(missing_markers),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -int(row["convertible_upper_bound"]),
            -int(row["missing_run_total"]),
            int(row["frame"]),
        )
    )
    suggested = choose_spaced(rows, args.top_n, args.min_frame_gap)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(args.report_dir / "manual_anchor_frame_candidates.csv", rows)
    base.write_csv(args.report_dir / "suggested_manual_anchor_frames.csv", suggested)

    print("Suggested manual frames:")
    if not suggested:
        print("  none")
    for row in suggested:
        print(
            "  rank={rank} frame={frame} score={score} missing={missing} grey={grey} missing_markers={markers}".format(
                rank=row["suggest_rank"],
                frame=row["frame"],
                score=row["score"],
                missing=row["missing_human_count"],
                grey=row["grey_point_count"],
                markers=row["missing_markers"],
            )
        )
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
