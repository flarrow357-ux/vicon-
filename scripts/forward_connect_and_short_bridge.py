from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base
import radius_bidirectional_rigid_connect as rigid


def find_short_gaps(valid_col: np.ndarray, start_i: int, end_i: int, max_gap: int):
    frame_i = start_i + 1
    while frame_i < end_i:
        if valid_col[frame_i]:
            frame_i += 1
            continue
        gap_start = frame_i
        while frame_i < end_i and not valid_col[frame_i]:
            frame_i += 1
        gap_end = frame_i - 1
        gap_len = gap_end - gap_start + 1
        if gap_len <= max_gap:
            yield gap_start, gap_end


def bridge_short_gaps(
    xyz,
    residual,
    valid,
    marker_names,
    label_to_i,
    start_i,
    end_i,
    refs,
    marker_to_segment,
    max_gap,
    first_frame,
):
    accepted = []
    rejected = []

    for marker in marker_names:
        marker_i = label_to_i[marker]
        for gap_start, gap_end in find_short_gaps(valid[:, marker_i], start_i, end_i, max_gap):
            left_i = gap_start - 1
            right_i = gap_end + 1
            if left_i < start_i or right_i > end_i or not valid[left_i, marker_i] or not valid[right_i, marker_i]:
                rejected.append(
                    {
                        "marker": marker,
                        "gap_start": gap_start + first_frame,
                        "gap_end": gap_end + first_frame,
                        "reason": "not_bracketed_by_labeled_marker",
                    }
                )
                continue

            span = right_i - left_i
            avg_step = float(np.linalg.norm(xyz[right_i, marker_i] - xyz[left_i, marker_i]) / span)
            step_limit = base.marker_step_limit(marker) * 0.85
            if avg_step > step_limit:
                rejected.append(
                    {
                        "marker": marker,
                        "gap_start": gap_start + first_frame,
                        "gap_end": gap_end + first_frame,
                        "left_frame": left_i + first_frame,
                        "right_frame": right_i + first_frame,
                        "reason": "motion_too_fast",
                        "avg_step": avg_step,
                        "step_limit": step_limit,
                    }
                )
                continue

            pending = []
            for frame_i in range(gap_start, gap_end + 1):
                t = (frame_i - left_i) / span
                pos = xyz[left_i, marker_i] * (1.0 - t) + xyz[right_i, marker_i] * t
                ok, support, mean_error, details = rigid.rigid_check(
                    marker,
                    pos,
                    frame_i,
                    xyz,
                    valid,
                    label_to_i,
                    refs,
                    marker_to_segment,
                )
                if not ok or mean_error > 30.0:
                    rejected.append(
                        {
                            "marker": marker,
                            "gap_start": gap_start + first_frame,
                            "gap_end": gap_end + first_frame,
                            "frame": frame_i + first_frame,
                            "reason": "weak_rigid_support",
                            "support": support,
                            "mean_error": mean_error,
                            "details": ";".join(details),
                        }
                    )
                    pending = []
                    break
                pending.append((frame_i, pos, t, support, mean_error, details))

            for frame_i, pos, t, support, mean_error, details in pending:
                xyz[frame_i, marker_i] = pos
                residual[frame_i, marker_i] = residual[left_i, marker_i] * (1.0 - t) + residual[right_i, marker_i] * t
                valid[frame_i, marker_i] = True
                accepted.append(
                    {
                        "frame": frame_i + first_frame,
                        "marker": marker,
                        "gap_start": gap_start + first_frame,
                        "gap_end": gap_end + first_frame,
                        "left_frame": left_i + first_frame,
                        "right_frame": right_i + first_frame,
                        "gap_len": gap_end - gap_start + 1,
                        "avg_step": avg_step,
                        "support": support,
                        "mean_error": mean_error,
                        "details": ";".join(details),
                        "method": "forward_short_gap_bridge",
                    }
                )

    return accepted, rejected


def candidate_margin(row):
    second_distance = row.get("second_distance")
    if second_distance in ("", None):
        return 999.0
    return float(second_distance) - float(row["distance"])


def marker_forward_error_limit(marker, default_limit, c7_valid=False, lbhd_c7_max_mean_error=15.0):
    if marker == "LBHD" and c7_valid:
        return min(default_limit, lbhd_c7_max_mean_error)
    if marker in {"RBHD", "LBHD", "RFHD", "LFHD"}:
        return min(default_limit, 10.0)
    return default_limit


def marker_forward_radius(marker, default_radius, head_radius):
    if marker in {"RBHD", "LBHD", "RFHD", "LFHD"}:
        return head_radius
    return default_radius


def marker_context_radius(marker, default_radius, head_radius, c7_valid=False, lbhd_c7_radius=45.0):
    if marker == "LBHD" and c7_valid:
        return lbhd_c7_radius
    return marker_forward_radius(marker, default_radius, head_radius)


def search_forward_current_first(
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
        for frame_i in range(start_i + 1, end_i + 1):
            if valid_work[frame_i, marker_i]:
                anchor_pos = xyz_work[frame_i, marker_i].copy()
                anchor_frame = frame_i
                continue
            if anchor_pos is None:
                continue

            found = None
            searched = []
            for offset in range(0, max_search + 1):
                target_i = frame_i + offset
                if target_i > end_i:
                    break
                if valid_work[target_i, marker_i]:
                    break
                searched.append(target_i)
                raw_options = []
                c7_valid = bool(c7_i is not None and valid_work[target_i, c7_i])
                active_radius = marker_context_radius(marker, radius, head_radius, c7_valid, lbhd_c7_radius)
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
                if mean_error > marker_forward_error_limit(
                    marker, max_mean_error, c7_valid, lbhd_c7_max_mean_error
                ):
                    skipped.append({**row, "reason": "mean_error_above_marker_limit"})
                    continue
                if candidate_margin(row) < min_margin:
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
                        "reason": "no_acceptable_forward_candidate",
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
            accepted.append({**found, "method": "forward_current_first_grey_connect"})

    return xyz_work, residual_work, valid_work, accepted, skipped, logs


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
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--forward-max-mean-error", type=float, default=25.0)
    parser.add_argument("--forward-min-margin", type=float, default=30.0)
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
    xyz_out, residual_out, valid_out, forward_accepted_raw, forward_skipped, forward_logs = search_forward_current_first(
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
        args.forward_max_mean_error,
        args.forward_min_margin,
        args.lbhd_c7_radius,
        args.lbhd_c7_max_mean_error,
    )

    forward_accepted = []
    for row in forward_accepted_raw:
        forward_accepted.append(
            {
                "frame": row["frame"] + layout.first_frame,
                "marker": row["marker"],
                "raw": row["raw"],
                "anchor_frame": row["anchor_frame"] + layout.first_frame,
                "distance": row["distance"],
                "second_distance": row["second_distance"],
                "candidate_count": row["candidate_count"],
                "radius_limit": row["radius_limit"],
                "support": row["support"],
                "mean_error": row["mean_error"],
                "search_offset": row["search_offset"],
                "details": row["details"],
                "method": row["method"],
            }
        )
    forward_skipped = [
        {
            **row,
            "frame": row["frame"] + layout.first_frame if isinstance(row.get("frame"), int) else row.get("frame", ""),
            "anchor_frame": row["anchor_frame"] + layout.first_frame
            if isinstance(row.get("anchor_frame"), int)
            else row.get("anchor_frame", ""),
        }
        for row in forward_skipped
    ]
    forward_logs = [
        {
            **row,
            "frame": row["frame"] + layout.first_frame if isinstance(row.get("frame"), int) else row.get("frame", ""),
            "anchor_frame": row["anchor_frame"] + layout.first_frame
            if isinstance(row.get("anchor_frame"), int)
            else row.get("anchor_frame", ""),
        }
        for row in forward_logs
    ]

    bridge_accepted, bridge_rejected = bridge_short_gaps(
        xyz_out,
        residual_out,
        valid_out,
        marker_names,
        label_to_i,
        start_i,
        end_i,
        refs,
        marker_to_segment,
        args.max_gap,
        layout.first_frame,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "accepted_forward_only_radius.csv", forward_accepted)
    base.write_csv(args.report_dir / "skipped_forward_only_radius.csv", forward_skipped)
    base.write_csv(args.report_dir / "accepted_forward_short_gap_bridge.csv", bridge_accepted)
    base.write_csv(args.report_dir / "rejected_forward_short_gap_bridge.csv", bridge_rejected)
    base.write_csv(args.report_dir / "search_logs_forward_only_radius.csv", forward_logs)

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
                    "connected_or_filled": before_missing - after_missing,
                }
            )
    base.write_csv(args.report_dir / "missing_summary_forward_connect_and_bridge.csv", summary)
    print(f"forward_accepted={len(forward_accepted)}")
    print(f"forward_skipped={len(forward_skipped)}")
    print(f"bridge_accepted={len(bridge_accepted)}")
    print(f"bridge_rejected={len(bridge_rejected)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
