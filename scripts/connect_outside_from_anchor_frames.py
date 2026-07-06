from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base
import forward_connect_and_short_bridge as forward_rules
import radius_bidirectional_rigid_connect as rigid


def candidate_row(
    marker,
    raw,
    target_i,
    anchor_frame,
    distance,
    second_distance,
    raw_options,
    search_offset,
    active_radius,
):
    return {
        "frame": target_i,
        "marker": marker,
        "raw": raw,
        "anchor_frame": anchor_frame,
        "distance": distance,
        "second_distance": second_distance,
        "candidate_count": len(raw_options),
        "search_offset": search_offset,
        "radius_limit": active_radius,
    }


def find_candidate(
    marker,
    marker_i,
    frame_i,
    step,
    limit_i,
    anchor_pos,
    anchor_frame,
    xyz_work,
    valid_work,
    labels,
    unlabeled,
    refs,
    marker_to_segment,
    label_to_i,
    radius,
    head_radius,
    max_search,
    max_mean_error,
    min_margin,
    lbhd_c7_radius,
    lbhd_c7_max_mean_error,
):
    logs = []
    c7_i = label_to_i.get("C7")
    searched = []
    for offset in range(0, max_search + 1):
        target_i = frame_i + step * offset
        if step > 0 and target_i > limit_i:
            break
        if step < 0 and target_i < limit_i:
            break
        if valid_work[target_i, marker_i]:
            break
        searched.append(target_i)

        c7_valid = bool(c7_i is not None and valid_work[target_i, c7_i])
        active_radius = forward_rules.marker_context_radius(marker, radius, head_radius, c7_valid, lbhd_c7_radius)
        raw_options = []
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
        row = candidate_row(
            marker,
            raw,
            target_i,
            anchor_frame,
            distance,
            second_distance,
            raw_options,
            offset,
            active_radius,
        )
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
        row.update({"support": support, "mean_error": mean_error, "details": ";".join(details)})
        if not ok:
            logs.append({**row, "reason": "rigid_check_failed"})
            continue
        if mean_error > forward_rules.marker_forward_error_limit(
            marker, max_mean_error, c7_valid, lbhd_c7_max_mean_error
        ):
            logs.append({**row, "reason": "mean_error_above_marker_limit"})
            continue
        if forward_rules.candidate_margin(row) < min_margin:
            logs.append({**row, "reason": "ambiguous_competing_unlabeled"})
            continue
        return row, logs

    logs.append(
        {
            "frame": frame_i,
            "marker": marker,
            "anchor_frame": anchor_frame,
            "searched_frames": ";".join(str(frame) for frame in searched),
            "reason": "no_acceptable_candidate",
        }
    )
    return None, logs


def connect_direction(
    direction,
    xyz_work,
    residual_work,
    valid_work,
    labels,
    marker_names,
    anchor_i,
    limit_i,
    refs,
    marker_to_segment,
    label_to_i,
    radius,
    head_radius,
    max_search,
    max_mean_error,
    min_margin,
    lbhd_c7_radius,
    lbhd_c7_max_mean_error,
):
    step = 1 if direction == "forward_after_end" else -1
    unlabeled = [label for label in labels if label.startswith("*")]
    accepted = []
    skipped = []
    logs = []

    for marker in marker_names:
        marker_i = label_to_i[marker]
        if not valid_work[anchor_i, marker_i]:
            skipped.append(
                {
                    "marker": marker,
                    "anchor_frame": anchor_i,
                    "reason": "marker_missing_at_anchor_frame",
                    "direction": direction,
                }
            )
            continue

        anchor_pos = xyz_work[anchor_i, marker_i].copy()
        anchor_frame = anchor_i
        frame_iter = range(anchor_i + step, limit_i + step, step)
        for frame_i in frame_iter:
            if valid_work[frame_i, marker_i]:
                anchor_pos = xyz_work[frame_i, marker_i].copy()
                anchor_frame = frame_i
                continue

            found, new_logs = find_candidate(
                marker,
                marker_i,
                frame_i,
                step,
                limit_i,
                anchor_pos,
                anchor_frame,
                xyz_work,
                valid_work,
                labels,
                unlabeled,
                refs,
                marker_to_segment,
                label_to_i,
                radius,
                head_radius,
                max_search,
                max_mean_error,
                min_margin,
                lbhd_c7_radius,
                lbhd_c7_max_mean_error,
            )
            logs.extend({**row, "direction": direction} for row in new_logs)
            if found is None:
                continue

            target_i = int(found["frame"])
            raw_i = label_to_i[found["raw"]]
            if valid_work[target_i, marker_i] or not valid_work[target_i, raw_i]:
                skipped.append({**found, "direction": direction, "reason": "target_or_raw_no_longer_available"})
                continue

            xyz_work[target_i, marker_i] = xyz_work[target_i, raw_i]
            residual_work[target_i, marker_i] = residual_work[target_i, raw_i]
            xyz_work[target_i, raw_i] = 0.0
            residual_work[target_i, raw_i] = -1.0
            valid_work[target_i, marker_i] = True
            valid_work[target_i, raw_i] = False
            anchor_pos = xyz_work[target_i, marker_i].copy()
            anchor_frame = target_i
            accepted.append({**found, "direction": direction, "method": "outside_anchor_grey_connect"})

    return accepted, skipped, logs


def externalize(rows, first_frame):
    out = []
    for row in rows:
        converted = dict(row)
        if isinstance(converted.get("frame"), int):
            converted["frame"] += first_frame
        if isinstance(converted.get("anchor_frame"), int):
            converted["anchor_frame"] += first_frame
        if "searched_frames" in converted and converted["searched_frames"]:
            converted["searched_frames"] = ";".join(
                str(int(frame) + first_frame) for frame in str(converted["searched_frames"]).split(";") if frame
            )
        out.append(converted)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-anchor-frame", type=int, required=True)
    parser.add_argument("--end-anchor-frame", type=int, required=True)
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
    start_anchor_i = args.start_anchor_frame - layout.first_frame
    end_anchor_i = args.end_anchor_frame - layout.first_frame
    for frame_i, name in [(start_anchor_i, "start_anchor"), (end_anchor_i, "end_anchor")]:
        missing_markers = [marker for marker in marker_names if not valid[frame_i, label_to_i[marker]]]
        if missing_markers:
            raise ValueError(f"{name} frame is not complete: {missing_markers}")

    xyz_work = xyz.copy()
    residual_work = residual.copy()
    valid_work = valid.copy()
    marker_to_segment = rigid.build_marker_to_segment()
    refs_start = rigid.build_reference_distances(xyz_work, valid_work, label_to_i, start_anchor_i)
    refs_end = rigid.build_reference_distances(xyz_work, valid_work, label_to_i, end_anchor_i)

    accepted_before, skipped_before, logs_before = connect_direction(
        "reverse_before_start",
        xyz_work,
        residual_work,
        valid_work,
        layout.labels,
        marker_names,
        start_anchor_i,
        0,
        refs_start,
        marker_to_segment,
        label_to_i,
        args.radius,
        args.head_radius,
        args.max_search,
        args.max_mean_error,
        args.min_margin,
        args.lbhd_c7_radius,
        args.lbhd_c7_max_mean_error,
    )
    accepted_after, skipped_after, logs_after = connect_direction(
        "forward_after_end",
        xyz_work,
        residual_work,
        valid_work,
        layout.labels,
        marker_names,
        end_anchor_i,
        layout.last_frame - layout.first_frame,
        refs_end,
        marker_to_segment,
        label_to_i,
        args.radius,
        args.head_radius,
        args.max_search,
        args.max_mean_error,
        args.min_margin,
        args.lbhd_c7_radius,
        args.lbhd_c7_max_mean_error,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_work, residual_work, args.output)
    base.write_csv(args.report_dir / "accepted_outside_anchor_connect.csv", externalize(accepted_before + accepted_after, layout.first_frame))
    base.write_csv(args.report_dir / "skipped_outside_anchor_connect.csv", externalize(skipped_before + skipped_after, layout.first_frame))
    base.write_csv(args.report_dir / "logs_outside_anchor_connect.csv", externalize(logs_before + logs_after, layout.first_frame))

    after_valid = base.is_valid(xyz_work, residual_work)
    summary = []
    for marker in marker_names:
        marker_i = label_to_i[marker]
        before_missing = int(
            np.sum(~valid[:start_anchor_i, marker_i]) + np.sum(~valid[end_anchor_i + 1 :, marker_i])
        )
        after_missing = int(
            np.sum(~after_valid[:start_anchor_i, marker_i])
            + np.sum(~after_valid[end_anchor_i + 1 :, marker_i])
        )
        if before_missing != after_missing:
            summary.append(
                {
                    "marker": marker,
                    "outside_missing_before": before_missing,
                    "outside_missing_after": after_missing,
                    "connected": before_missing - after_missing,
                }
            )
    base.write_csv(args.report_dir / "missing_summary_outside_anchor_connect.csv", summary)
    print(f"accepted_before={len(accepted_before)}")
    print(f"accepted_after={len(accepted_after)}")
    print(f"skipped={len(skipped_before) + len(skipped_after)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
