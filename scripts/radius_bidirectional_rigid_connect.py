from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base


SEGMENTS = {
    "head": ["RBHD", "LBHD", "RFHD", "LFHD"],
    "trunk": ["CLAV", "C7", "STRN", "T10"],
    "left_upper_arm": ["LSHO", "LUP", "LELB", "LELA"],
    "left_forearm_hand": ["LDW", "LWRB", "LFIN", "LWRA"],
    "right_upper_arm": ["RSHO", "RUP", "RELB", "RELA"],
    "right_forearm_hand": ["RDW", "RWRB", "RFIN", "RWRA"],
    "pelvis": ["RASIS", "LASIS", "RPSIS", "LPSIS"],
    "left_thigh": ["LTROC", "LTH", "LMEP", "LLEP"],
    "left_shank": ["LSK", "LLME", "LMME"],
    "left_foot": ["LHM2", "LHM5", "LHEEL", "LHM1"],
    "right_thigh": ["RTROC", "RTH", "RMEP", "RLEP"],
    "right_shank": ["RSK", "RLME", "RMME"],
    "right_foot": ["RHM2", "RHM1", "RHEEL", "RHM5"],
}


def build_marker_to_segment():
    out = {}
    for segment, markers in SEGMENTS.items():
        for marker in markers:
            out[marker] = segment
    return out


def build_reference_distances(xyz, valid, label_to_i, start_i):
    refs = {}
    for segment, markers in SEGMENTS.items():
        for marker in markers:
            if marker not in label_to_i:
                continue
            marker_i = label_to_i[marker]
            for other in markers:
                if other == marker or other not in label_to_i:
                    continue
                other_i = label_to_i[other]
                if valid[start_i, marker_i] and valid[start_i, other_i]:
                    refs[(marker, other)] = float(np.linalg.norm(xyz[start_i, marker_i] - xyz[start_i, other_i]))
    return refs


def rigid_check(marker, candidate_pos, frame_i, xyz, valid, label_to_i, refs, marker_to_segment):
    segment = marker_to_segment.get(marker)
    if not segment:
        return False, 0, 999.0, []
    errors = []
    support = 0
    details = []
    for other in SEGMENTS[segment]:
        if other == marker or other not in label_to_i:
            continue
        other_i = label_to_i[other]
        if not valid[frame_i, other_i]:
            continue
        ref = refs.get((marker, other))
        if ref is None:
            continue
        dist = float(np.linalg.norm(candidate_pos - xyz[frame_i, other_i]))
        err = abs(dist - ref)
        tolerance = max(25.0, min(55.0, ref * 0.18))
        ok = err <= tolerance
        support += 1 if ok else 0
        errors.append(err)
        details.append(f"{other}:{dist:.1f}/{ref:.1f}/err{err:.1f}")
    segment_size = len(SEGMENTS[segment])
    required = 2 if segment_size >= 4 else 1
    mean_error = float(np.mean(errors)) if errors else 999.0
    return support >= required and mean_error <= 35.0, support, mean_error, details


def is_high_conf_single(row, max_distance=15.0, max_mean_error=5.0, min_support=2, min_margin=30.0):
    second_distance = row.get("second_distance")
    if second_distance in ("", None):
        distance_margin = 999.0
    else:
        distance_margin = float(second_distance) - float(row["distance"])
    return (
        float(row["distance"]) <= max_distance
        and float(row["mean_error"]) <= max_mean_error
        and int(row["support"]) >= min_support
        and distance_margin >= min_margin
    )


def search_direction(
    direction,
    xyz,
    residual,
    valid,
    labels,
    marker_names,
    start_i,
    end_i,
    radius,
    max_search,
    refs,
    marker_to_segment,
    label_to_i,
):
    xyz_work = xyz.copy()
    residual_work = residual.copy()
    valid_work = valid.copy()
    unlabeled = [label for label in labels if label.startswith("*")]
    candidates = {}
    logs = []

    frame_range = range(start_i + 1, end_i + 1) if direction == "forward" else range(end_i - 1, start_i - 1, -1)
    step = 1 if direction == "forward" else -1

    for marker in marker_names:
        marker_i = label_to_i[marker]
        anchor_pos = None
        anchor_frame = None
        for frame_i in frame_range:
            if valid_work[frame_i, marker_i]:
                anchor_pos = xyz_work[frame_i, marker_i].copy()
                anchor_frame = frame_i
                continue
            if anchor_pos is None:
                continue

            found = None
            search_frames = []
            for offset in range(1, max_search + 1):
                target_i = frame_i + step * offset
                if target_i < start_i or target_i > end_i:
                    break
                search_frames.append(target_i)
                raw_options = []
                for raw in unlabeled:
                    raw_i = label_to_i[raw]
                    if not valid_work[target_i, raw_i]:
                        continue
                    distance = float(np.linalg.norm(xyz_work[target_i, raw_i] - anchor_pos))
                    if distance <= radius:
                        raw_options.append((distance, raw, raw_i))
                if not raw_options:
                    continue
                raw_options.sort(key=lambda item: item[0])
                distance, raw, raw_i = raw_options[0]
                second_distance = raw_options[1][0] if len(raw_options) > 1 else ""
                ok, support, mean_error, details = rigid_check(
                    marker,
                    xyz_work[target_i, raw_i],
                    target_i,
                    xyz_work,
                    valid_work,
                    label_to_i,
                    refs,
                    marker_to_segment,
                )
                if not ok:
                    logs.append(
                        {
                            "direction": direction,
                            "frame": target_i,
                            "marker": marker,
                            "raw": raw,
                            "anchor_frame": anchor_frame,
                            "distance": distance,
                            "second_distance": second_distance,
                            "candidate_count": len(raw_options),
                            "support": support,
                            "mean_error": mean_error,
                            "reason": "rigid_check_failed",
                            "details": ";".join(details),
                        }
                    )
                    continue
                found = {
                    "direction": direction,
                    "frame": target_i,
                    "marker": marker,
                    "raw": raw,
                    "anchor_frame": anchor_frame,
                    "distance": distance,
                    "second_distance": second_distance,
                    "candidate_count": len(raw_options),
                    "support": support,
                    "mean_error": mean_error,
                    "search_offset": offset,
                    "details": ";".join(details),
                }
                break

            if found is None:
                logs.append(
                    {
                        "direction": direction,
                        "frame": frame_i,
                        "marker": marker,
                        "anchor_frame": anchor_frame,
                        "search_frames": ";".join(str(f) for f in search_frames),
                        "reason": "no_candidate_within_radius",
                    }
                )
                continue

            target_i = int(found["frame"])
            raw_i = label_to_i[found["raw"]]
            if valid_work[target_i, marker_i] or not valid_work[target_i, raw_i]:
                continue
            xyz_work[target_i, marker_i] = xyz_work[target_i, raw_i]
            residual_work[target_i, marker_i] = residual_work[target_i, raw_i]
            xyz_work[target_i, raw_i] = 0.0
            residual_work[target_i, raw_i] = -1.0
            valid_work[target_i, marker_i] = True
            valid_work[target_i, raw_i] = False
            anchor_pos = xyz_work[target_i, marker_i].copy()
            anchor_frame = target_i
            candidates[(target_i, marker)] = found

    return candidates, logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--max-search", type=int, default=5)
    parser.add_argument("--single-high-conf-distance", type=float, default=15.0)
    parser.add_argument("--single-high-conf-mean-error", type=float, default=5.0)
    parser.add_argument("--single-high-conf-min-support", type=int, default=2)
    parser.add_argument("--single-high-conf-min-margin", type=float, default=30.0)
    args = parser.parse_args()

    data, layout, xyz, residual = base.load_c3d(args.c3d)
    marker_names, _edges = base.load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    valid = base.is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame

    missing_at_start = [marker for marker in marker_names if not valid[start_i, label_to_i[marker]]]
    missing_at_end = [marker for marker in marker_names if not valid[end_i, label_to_i[marker]]]
    if missing_at_start or missing_at_end:
        raise ValueError(f"Anchor frames must be complete. start_missing={missing_at_start}, end_missing={missing_at_end}")

    marker_to_segment = build_marker_to_segment()
    refs = build_reference_distances(xyz, valid, label_to_i, start_i)
    forward, forward_logs = search_direction(
        "forward",
        xyz,
        residual,
        valid,
        layout.labels,
        marker_names,
        start_i,
        end_i,
        args.radius,
        args.max_search,
        refs,
        marker_to_segment,
        label_to_i,
    )
    backward, backward_logs = search_direction(
        "backward",
        xyz,
        residual,
        valid,
        layout.labels,
        marker_names,
        start_i,
        end_i,
        args.radius,
        args.max_search,
        refs,
        marker_to_segment,
        label_to_i,
    )

    accepted = []
    disagreed = []
    forward_only = []
    backward_only = []
    for key in sorted(set(forward) | set(backward)):
        f = forward.get(key)
        b = backward.get(key)
        frame_i, marker = key
        if f and b and f["raw"] == b["raw"]:
            accepted.append(
                {
                    "frame": frame_i + layout.first_frame,
                    "marker": marker,
                    "raw": f["raw"],
                    "forward_distance": f["distance"],
                    "backward_distance": b["distance"],
                    "forward_anchor_frame": f["anchor_frame"] + layout.first_frame,
                    "backward_anchor_frame": b["anchor_frame"] + layout.first_frame,
                    "support": min(f["support"], b["support"]),
                    "mean_error": max(f["mean_error"], b["mean_error"]),
                    "method": "radius_bidirectional_rigid",
                }
            )
        elif f and b:
            disagreed.append(
                {
                    "frame": frame_i + layout.first_frame,
                    "marker": marker,
                    "forward_raw": f["raw"],
                    "backward_raw": b["raw"],
                    "reason": "bidirectional_disagreement",
                }
            )
        elif f:
            row = {**f, "frame": frame_i + layout.first_frame}
            if is_high_conf_single(
                f,
                args.single_high_conf_distance,
                args.single_high_conf_mean_error,
                args.single_high_conf_min_support,
                args.single_high_conf_min_margin,
            ):
                accepted.append(
                    {
                        "frame": frame_i + layout.first_frame,
                        "marker": marker,
                        "raw": f["raw"],
                        "forward_distance": f["distance"],
                        "backward_distance": "",
                        "forward_anchor_frame": f["anchor_frame"] + layout.first_frame,
                        "backward_anchor_frame": "",
                        "support": f["support"],
                        "mean_error": f["mean_error"],
                        "second_distance": f["second_distance"],
                        "candidate_count": f["candidate_count"],
                        "method": "radius_single_high_conf_forward",
                    }
                )
            else:
                forward_only.append({**row, "reason": "forward_only"})
        elif b:
            row = {**b, "frame": frame_i + layout.first_frame}
            if is_high_conf_single(
                b,
                args.single_high_conf_distance,
                args.single_high_conf_mean_error,
                args.single_high_conf_min_support,
                args.single_high_conf_min_margin,
            ):
                accepted.append(
                    {
                        "frame": frame_i + layout.first_frame,
                        "marker": marker,
                        "raw": b["raw"],
                        "forward_distance": "",
                        "backward_distance": b["distance"],
                        "forward_anchor_frame": "",
                        "backward_anchor_frame": b["anchor_frame"] + layout.first_frame,
                        "support": b["support"],
                        "mean_error": b["mean_error"],
                        "second_distance": b["second_distance"],
                        "candidate_count": b["candidate_count"],
                        "method": "radius_single_high_conf_backward",
                    }
                )
            else:
                backward_only.append({**row, "reason": "backward_only"})

    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    for row in accepted:
        frame_i = int(row["frame"]) - layout.first_frame
        marker_i = label_to_i[row["marker"]]
        raw_i = label_to_i[row["raw"]]
        if valid_out[frame_i, marker_i] or not valid_out[frame_i, raw_i]:
            continue
        xyz_out[frame_i, marker_i] = xyz_out[frame_i, raw_i]
        residual_out[frame_i, marker_i] = residual_out[frame_i, raw_i]
        xyz_out[frame_i, raw_i] = 0.0
        residual_out[frame_i, raw_i] = -1.0
        valid_out[frame_i, marker_i] = True
        valid_out[frame_i, raw_i] = False

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "accepted_radius_bidirectional.csv", accepted)
    base.write_csv(args.report_dir / "disagreed_radius_bidirectional.csv", disagreed)
    base.write_csv(args.report_dir / "forward_only_radius_bidirectional.csv", forward_only)
    base.write_csv(args.report_dir / "backward_only_radius_bidirectional.csv", backward_only)
    base.write_csv(args.report_dir / "search_logs_radius_bidirectional.csv", forward_logs + backward_logs)

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
    base.write_csv(args.report_dir / "missing_summary_radius_bidirectional.csv", summary)
    print(f"accepted={len(accepted)}")
    print(f"disagreed={len(disagreed)}")
    print(f"forward_only={len(forward_only)}")
    print(f"backward_only={len(backward_only)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
