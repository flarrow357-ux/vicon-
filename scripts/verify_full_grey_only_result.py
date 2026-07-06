from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base


def find_same_frame_raw(pos, raw_items, used):
    for idx, (_label, raw_pos) in enumerate(raw_items):
        if idx in used:
            continue
        if np.allclose(pos, raw_pos, atol=1e-3, rtol=0.0):
            used.add(idx)
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-c3d", required=True, type=Path)
    parser.add_argument("--final-c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()

    _orig_data, orig_layout, orig_xyz, orig_residual = base.load_c3d(args.original_c3d)
    _final_data, final_layout, final_xyz, final_residual = base.load_c3d(args.final_c3d)
    marker_names, _edges = base.load_model(args.model)

    if orig_layout.first_frame != final_layout.first_frame or orig_layout.last_frame != final_layout.last_frame:
        raise ValueError("Original and final C3D frame ranges differ.")
    if orig_layout.labels != final_layout.labels:
        raise ValueError("Original and final C3D labels differ.")

    label_to_i = {label: i for i, label in enumerate(final_layout.labels)}
    missing_model_markers = [marker for marker in marker_names if marker not in label_to_i]
    if missing_model_markers:
        raise ValueError(f"Model markers missing from C3D labels: {missing_model_markers}")

    valid_orig = base.is_valid(orig_xyz, orig_residual)
    valid_final = base.is_valid(final_xyz, final_residual)

    raw_labels = [label for label in final_layout.labels if label.startswith("*")]
    raw_by_frame = {}
    removed_raw_rows = []
    point_count_change_rows = []

    for frame_i in range(valid_orig.shape[0]):
        original_valid_points = int(np.sum(valid_orig[frame_i]))
        final_valid_points = int(np.sum(valid_final[frame_i]))
        if original_valid_points != final_valid_points:
            point_count_change_rows.append(
                {
                    "frame": frame_i + final_layout.first_frame,
                    "original_valid_points": original_valid_points,
                    "final_valid_points": final_valid_points,
                    "delta": final_valid_points - original_valid_points,
                }
            )

        items = []
        for label in raw_labels:
            point_i = label_to_i[label]
            if valid_orig[frame_i, point_i] and not valid_final[frame_i, point_i]:
                pos = orig_xyz[frame_i, point_i].copy()
                items.append((label, pos))
                removed_raw_rows.append(
                    {
                        "frame": frame_i + final_layout.first_frame,
                        "raw": label,
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2]),
                    }
                )
        raw_by_frame[frame_i] = items

    new_human_rows = []
    not_from_raw_rows = []
    for frame_i in range(valid_orig.shape[0]):
        used = set()
        for marker in marker_names:
            marker_i = label_to_i[marker]
            if valid_orig[frame_i, marker_i] or not valid_final[frame_i, marker_i]:
                continue
            pos = final_xyz[frame_i, marker_i].copy()
            row = {
                "frame": frame_i + final_layout.first_frame,
                "marker": marker,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }
            new_human_rows.append(row)
            if not find_same_frame_raw(pos, raw_by_frame[frame_i], used):
                not_from_raw_rows.append(row)

    start_complete = None
    end_complete = None
    if args.start_frame is not None:
        start_i = args.start_frame - final_layout.first_frame
        start_complete = all(valid_final[start_i, label_to_i[marker]] for marker in marker_names)
    if args.end_frame is not None:
        end_i = args.end_frame - final_layout.first_frame
        end_complete = all(valid_final[end_i, label_to_i[marker]] for marker in marker_names)

    new_by_marker = Counter(row["marker"] for row in new_human_rows)
    summary = {
        "original_c3d": str(args.original_c3d),
        "final_c3d": str(args.final_c3d),
        "new_human_points": len(new_human_rows),
        "removed_raw_points": len(removed_raw_rows),
        "not_from_same_frame_raw": len(not_from_raw_rows),
        "frames_with_point_count_change": len(point_count_change_rows),
        "start_complete": start_complete,
        "end_complete": end_complete,
        "new_by_marker": dict(sorted(new_by_marker.items())),
        "passed": (
            len(not_from_raw_rows) == 0
            and len(point_count_change_rows) == 0
            and len(new_human_rows) == len(removed_raw_rows)
            and (start_complete is not False)
            and (end_complete is not False)
        ),
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(args.report_dir / "verify_full_new_human_points.csv", new_human_rows)
    base.write_csv(args.report_dir / "verify_full_removed_raw_points.csv", removed_raw_rows)
    base.write_csv(args.report_dir / "verify_full_not_from_same_frame_raw.csv", not_from_raw_rows)
    base.write_csv(args.report_dir / "verify_full_point_count_changes.csv", point_count_change_rows)
    (args.report_dir / "verify_full_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
