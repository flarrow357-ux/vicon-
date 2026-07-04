from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base
import forward_connect_and_short_bridge as forward_rules
import radius_bidirectional_rigid_connect as rigid


def search_reverse_current_first(
    xyz,
    residual,
    valid,
    labels,
    marker_names,
    start_i,
    end_i,
    radius,
    head_radius,
    max_search,
    refs,
    marker_to_segment,
    label_to_i,
    max_mean_error,
    min_margin,
    lbhd_c7_radius,
    lbhd_c7_max_mean_error,
):
    xyz_work = xyz.copy()
    residual_work = residual.copy()
    valid_work = valid.copy()
    unlabeled = [label for label in labels if label.startswith("*")]
    accepted = []
    skipped = []
    logs = []

    for marker in marker_names:
        marker_i = label_to_i[marker]
        c7_i = label_to_i.get("C7")
        anchor_pos = None
        anchor_frame = None
        for frame_i in range(end_i - 1, start_i - 1, -1):
            if valid_work[frame_i, marker_i]:
                anchor_pos = xyz_work[frame_i, marker_i].copy()
                anchor_frame = frame_i
                continue
            if anchor_pos is None:
                continue

            found = None
            searched = []
            for offset in range(0, max_search + 1):
                target_i = frame_i - offset
                if target_i < start_i:
                    break
                if valid_work[target_i, marker_i]:
                    break
                searched.append(target_i)
                raw_options = []
                c7_valid = bool(c7_i is not None and valid_work[target_i, c7_i])
                active_radius = forward_rules.marker_context_radius(
                    marker, radius, head_radius, c7_valid, lbhd_c7_radius
                )
                for raw in unlabeled:
                    raw_i = label_to_i[raw]
                    if not valid_work[target_i, raw_i]:
                        continue
                    distance = float(np.linalg.norm(xyz_work[target_i, raw_i] - anchor_pos))
                    if distance <= active_radius:
                        raw_options.append((distance, raw, raw_i))
                if not raw_options:
                    continue

                raw_options.sort(key=lambda item: item[0])
                distance, raw, raw_i = raw_options[0]
                second_distance = raw_options[1][0] if len(raw_options) > 1 else ""
                row = {
                    "frame": target_i,
                    "marker": marker,
                    "raw": raw,
                    "anchor_frame": anchor_frame,
                    "distance": distance,
                    "second_distance": second_distance,
                    "candidate_count": len(raw_options),
                    "search_offset": offset,
                    "radius_limit": active_radius,
                }
                ok, support, mean_error, details = rigid.rigid_check(
                    marker,
                    xyz_work[target_i, raw_i],
                    target_i,
                    xyz_work,
                    valid_work,
                    label_to_i,
                    refs,
                    marker_to_segment,
                )
                row.update(
                    {
                        "support": support,
                        "mean_error": mean_error,
                        "details": ";".join(details),
                    }
                )
                if not ok:
                    logs.append({**row, "reason": "rigid_check_failed"})
                    continue
                if mean_error > forward_rules.marker_forward_error_limit(
                    marker, max_mean_error, c7_valid, lbhd_c7_max_mean_error
                ):
                    skipped.append({**row, "reason": "mean_error_above_marker_limit"})
                    continue
                if forward_rules.candidate_margin(row) < min_margin:
                    skipped.append({**row, "reason": "ambiguous_competing_unlabeled"})
                    continue
                found = row
                break

            if found is None:
                logs.append(
                    {
                        "frame": frame_i,
                        "marker": marker,
                        "anchor_frame": anchor_frame,
                        "searched_frames": ";".join(str(f) for f in searched),
                        "reason": "no_acceptable_reverse_candidate",
                    }
                )
                continue

            target_i = int(found["frame"])
            raw_i = label_to_i[found["raw"]]
            if valid_work[target_i, marker_i] or not valid_work[target_i, raw_i]:
                skipped.append({**found, "reason": "target_or_raw_no_longer_available"})
                continue
            xyz_work[target_i, marker_i] = xyz_work[target_i, raw_i]
            residual_work[target_i, marker_i] = residual_work[target_i, raw_i]
            xyz_work[target_i, raw_i] = 0.0
            residual_work[target_i, raw_i] = -1.0
            valid_work[target_i, marker_i] = True
            valid_work[target_i, raw_i] = False
            anchor_pos = xyz_work[target_i, marker_i].copy()
            anchor_frame = target_i
            accepted.append({**found, "method": "reverse_current_first_grey_connect"})

    return xyz_work, residual_work, valid_work, accepted, skipped, logs


def to_external_rows(rows, first_frame):
    out = []
    for row in rows:
        converted = dict(row)
        if isinstance(converted.get("frame"), int):
            converted["frame"] = converted["frame"] + first_frame
        if isinstance(converted.get("anchor_frame"), int):
            converted["anchor_frame"] = converted["anchor_frame"] + first_frame
        out.append(converted)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--radius", type=float, default=60.0)
    parser.add_argument("--head-radius", type=float, default=25.0)
    parser.add_argument("--max-search", type=int, default=5)
    parser.add_argument("--max-mean-error", type=float, default=25.0)
    parser.add_argument("--min-margin", type=float, default=30.0)
    parser.add_argument("--lbhd-c7-radius", type=float, default=45.0)
    parser.add_argument("--lbhd-c7-max-mean-error", type=float, default=15.0)
    args = parser.parse_args()

    data, layout, xyz, residual = base.load_c3d(args.c3d)
    marker_names, _edges = base.load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    missing = [marker for marker in marker_names if marker not in label_to_i]
    if missing:
        raise ValueError(f"Model markers missing from C3D labels: {missing}")

    valid = base.is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame
    missing_at_start = [marker for marker in marker_names if not valid[start_i, label_to_i[marker]]]
    missing_at_end = [marker for marker in marker_names if not valid[end_i, label_to_i[marker]]]
    if missing_at_start or missing_at_end:
        raise ValueError(f"Anchor frames must be complete. start_missing={missing_at_start}, end_missing={missing_at_end}")

    marker_to_segment = rigid.build_marker_to_segment()
    refs = rigid.build_reference_distances(xyz, valid, label_to_i, start_i)
    xyz_out, residual_out, valid_out, accepted, skipped, logs = search_reverse_current_first(
        xyz,
        residual,
        valid,
        layout.labels,
        marker_names,
        start_i,
        end_i,
        args.radius,
        args.head_radius,
        args.max_search,
        refs,
        marker_to_segment,
        label_to_i,
        args.max_mean_error,
        args.min_margin,
        args.lbhd_c7_radius,
        args.lbhd_c7_max_mean_error,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "accepted_reverse_grey_only.csv", to_external_rows(accepted, layout.first_frame))
    base.write_csv(args.report_dir / "skipped_reverse_grey_only.csv", to_external_rows(skipped, layout.first_frame))
    base.write_csv(args.report_dir / "search_logs_reverse_grey_only.csv", to_external_rows(logs, layout.first_frame))

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
                    "connected": before_missing - after_missing,
                }
            )
    base.write_csv(args.report_dir / "missing_summary_reverse_grey_only.csv", summary)
    print(f"reverse_accepted={len(accepted)}")
    print(f"reverse_skipped={len(skipped)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
