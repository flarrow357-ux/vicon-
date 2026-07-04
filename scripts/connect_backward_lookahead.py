from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import conservative_c3d_connect as base


def future_anchors(valid: np.ndarray, marker_i: int, frame_i: int, end_i: int, max_lookahead: int):
    anchors = []
    for step in range(1, max_lookahead + 1):
        candidate = frame_i + step
        if candidate <= end_i and valid[candidate, marker_i]:
            anchors.append(candidate)
        if len(anchors) >= 2:
            break
    return anchors


def predict_from_future(xyz: np.ndarray, marker_i: int, frame_i: int, anchors: list[int]):
    first = anchors[0]
    prediction = xyz[first, marker_i].copy()
    if len(anchors) >= 2:
        second = anchors[1]
        velocity = (xyz[second, marker_i] - xyz[first, marker_i]) / (second - first)
        prediction = xyz[first, marker_i] - velocity * (first - frame_i)
    return prediction


def pass_threshold(marker: str, info: dict):
    pred_dist = float(info["pred_dist"])
    seg_error = float(info["seg_error"])
    support = int(info["support"])
    margin = float(info["margin"])
    required = base.support_requirement(marker)

    if support < required or margin < 80.0:
        return False
    if support >= 3 and pred_dist <= 30.0 and seg_error <= 25.0:
        return True
    if support >= 3 and pred_dist <= 55.0 and seg_error <= 8.0:
        return True
    if support == 2 and pred_dist <= 35.0 and seg_error <= 18.0:
        return True
    return False


def run_backward_lookahead(
    xyz: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
    layout: base.C3DLayout,
    marker_names: list[str],
    label_to_i: dict[str, int],
    refs: dict[tuple[str, str], float],
    neighbors: dict[str, list[str]],
    start_i: int,
    end_i: int,
    max_lookahead: int,
):
    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    accepted: list[dict] = []
    rejected: list[dict] = []
    unlabeled = [label for label in layout.labels if label.startswith("*")]

    for frame_i in range(end_i - 1, start_i, -1):
        grey = [label for label in unlabeled if valid_out[frame_i, label_to_i[label]]]
        if not grey:
            continue

        missing = []
        marker_predictions = {}
        for marker in marker_names:
            marker_i = label_to_i[marker]
            if valid_out[frame_i, marker_i]:
                continue
            anchors = future_anchors(valid_out, marker_i, frame_i, end_i, max_lookahead)
            if not anchors:
                continue
            missing.append(marker)
            marker_predictions[marker] = (
                predict_from_future(xyz_out, marker_i, frame_i, anchors),
                anchors,
            )

        if not missing:
            continue

        cost = np.full((len(missing), len(grey)), 1e6, dtype=float)
        meta: dict[tuple[int, int], dict] = {}
        for mi, marker in enumerate(missing):
            prediction, anchors = marker_predictions[marker]
            scored = []
            for gi, raw_label in enumerate(grey):
                raw_i = label_to_i[raw_label]
                result = base.candidate_score(
                    marker,
                    xyz_out[frame_i, raw_i],
                    prediction,
                    xyz_out,
                    valid_out,
                    frame_i,
                    refs,
                    neighbors,
                    label_to_i,
                )
                if result is None:
                    continue
                score, pred_dist, seg_error, support = result
                scored.append((score, gi, raw_label, pred_dist, seg_error, support))
            scored.sort(key=lambda item: item[0])
            if not scored:
                continue
            for rank, (score, gi, raw_label, pred_dist, seg_error, support) in enumerate(scored[:2]):
                margin = (scored[1][0] - score) if rank == 0 and len(scored) > 1 else 999.0
                if rank > 0:
                    continue
                info = {
                    "frame": frame_i + layout.first_frame,
                    "marker": marker,
                    "raw": raw_label,
                    "score": score,
                    "pred_dist": pred_dist,
                    "seg_error": seg_error,
                    "support": support,
                    "margin": margin,
                    "anchor_frames": ";".join(str(a + layout.first_frame) for a in anchors),
                    "lookahead": anchors[0] - frame_i,
                    "method": "backward_lookahead_connect",
                }
                if pass_threshold(marker, info):
                    cost[mi, gi] = score
                    meta[(mi, gi)] = info
                else:
                    rejected.append({**info, "reason": "threshold"})

        row_ind, col_ind = linear_sum_assignment(cost)
        for row, col in zip(row_ind, col_ind):
            if cost[row, col] >= 1e6:
                continue
            info = meta[(int(row), int(col))]
            marker_i = label_to_i[info["marker"]]
            raw_i = label_to_i[info["raw"]]
            if valid_out[frame_i, marker_i] or not valid_out[frame_i, raw_i]:
                continue
            xyz_out[frame_i, marker_i] = xyz_out[frame_i, raw_i]
            residual_out[frame_i, marker_i] = residual_out[frame_i, raw_i]
            xyz_out[frame_i, raw_i] = 0.0
            residual_out[frame_i, raw_i] = -1.0
            valid_out[frame_i, marker_i] = True
            valid_out[frame_i, raw_i] = False
            accepted.append(info)

    return xyz_out, residual_out, accepted, rejected


def split_segments(rows: list[dict]):
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["marker"], row["raw"]), []).append(row)

    accepted: list[dict] = []
    held: list[dict] = []
    for (_marker, _raw), items in grouped.items():
        items.sort(key=lambda row: int(row["frame"]))
        segment: list[dict] = []

        def flush():
            if not segment:
                return
            target = accepted if len(segment) <= 25 else held
            reason = "" if len(segment) <= 25 else "long_continuous_segment_needs_manual_check"
            for item in segment:
                if reason:
                    item = {**item, "reason": reason, "segment_len": len(segment)}
                else:
                    item = {**item, "segment_len": len(segment)}
                target.append(item)

        previous = None
        for item in items:
            frame = int(item["frame"])
            if previous is not None and frame != previous + 1:
                flush()
                segment = []
            segment.append(item)
            previous = frame
        flush()
    accepted.sort(key=lambda row: (int(row["frame"]), row["marker"]))
    held.sort(key=lambda row: (row["marker"], row["raw"], int(row["frame"])))
    return accepted, held


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=2127)
    parser.add_argument("--end-frame", type=int, default=3041)
    parser.add_argument("--max-lookahead", type=int, default=3)
    args = parser.parse_args()

    data, layout, xyz, residual = base.load_c3d(args.c3d)
    marker_names, edges = base.load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    valid = base.is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame

    refs = base.build_reference_lengths(xyz, valid, label_to_i, edges, start_i, end_i)
    neighbors = base.neighbor_map(edges)
    xyz_candidate, residual_candidate, accepted, rejected = run_backward_lookahead(
        xyz,
        residual,
        valid,
        layout,
        marker_names,
        label_to_i,
        refs,
        neighbors,
        start_i,
        end_i,
        args.max_lookahead,
    )
    accepted, held_for_review = split_segments(accepted)

    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    for item in accepted:
        frame_i = int(item["frame"]) - layout.first_frame
        marker_i = label_to_i[item["marker"]]
        raw_i = label_to_i[item["raw"]]
        if valid_out[frame_i, marker_i] or not valid_out[frame_i, raw_i]:
            continue
        xyz_out[frame_i, marker_i] = xyz[frame_i, raw_i]
        residual_out[frame_i, marker_i] = residual[frame_i, raw_i]
        xyz_out[frame_i, raw_i] = 0.0
        residual_out[frame_i, raw_i] = -1.0
        valid_out[frame_i, marker_i] = True
        valid_out[frame_i, raw_i] = False

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "accepted_backward_lookahead.csv", accepted)
    base.write_csv(args.report_dir / "held_long_segments_backward_lookahead.csv", held_for_review)
    base.write_csv(args.report_dir / "rejected_backward_lookahead.csv", rejected)

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
    base.write_csv(args.report_dir / "missing_summary_after_backward_lookahead.csv", summary)

    print(f"accepted={len(accepted)}")
    print(f"held_long_segments={len(held_for_review)}")
    print(f"rejected={len(rejected)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
