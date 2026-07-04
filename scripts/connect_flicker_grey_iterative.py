from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base


def marker_gaps(valid_col: np.ndarray, start_i: int, end_i: int):
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


def local_model_score(
    marker: str,
    candidate: np.ndarray,
    prediction: np.ndarray,
    xyz: np.ndarray,
    valid: np.ndarray,
    frame_i: int,
    label_to_i: dict[str, int],
    refs: dict[tuple[str, str], float],
    neighbors: dict[str, list[str]],
):
    pred_dist = float(np.linalg.norm(candidate - prediction))
    support = 0
    errors = []
    for neighbor in neighbors.get(marker, []):
        neighbor_i = label_to_i[neighbor]
        if not valid[frame_i, neighbor_i]:
            continue
        ref = refs.get((marker, neighbor))
        if ref is None:
            continue
        error = abs(float(np.linalg.norm(candidate - xyz[frame_i, neighbor_i])) - ref)
        errors.append(error)
        tolerance = max(45.0, min(95.0, ref * 0.32))
        if error <= tolerance:
            support += 1
    mean_error = float(np.mean(errors)) if errors else 999.0
    return pred_dist, support, mean_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=2127)
    parser.add_argument("--end-frame", type=int, default=3041)
    parser.add_argument("--max-gap", type=int, default=80)
    parser.add_argument("--max-bracket-span", type=int, default=120)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-frames-per-gap-pass", type=int, default=6)
    args = parser.parse_args()

    data, layout, xyz, residual = base.load_c3d(args.c3d)
    marker_names, edges = base.load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    valid = base.is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame
    refs = base.build_reference_lengths(xyz, valid, label_to_i, edges, start_i, end_i)
    neighbors = base.neighbor_map(edges)
    unlabeled = [label for label in layout.labels if label.startswith("*")]

    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    accepted: list[dict] = []
    rejected: list[dict] = []

    for iteration in range(1, args.max_iterations + 1):
        iteration_rows = []
        for marker in marker_names:
            marker_i = label_to_i[marker]
            for gap_start, gap_end in marker_gaps(valid_out[:, marker_i], start_i, end_i):
                gap_len = gap_end - gap_start + 1
                if gap_len > args.max_gap:
                    continue
                left_i = None
                right_i = None
                for frame_i in range(gap_start - 1, start_i - 1, -1):
                    if valid_out[frame_i, marker_i]:
                        left_i = frame_i
                        break
                for frame_i in range(gap_end + 1, end_i + 1):
                    if valid_out[frame_i, marker_i]:
                        right_i = frame_i
                        break
                if left_i is None or right_i is None or right_i - left_i > args.max_bracket_span:
                    continue

                candidates = []
                # Start from the later end of the gap and move backward; this targets flickering grey points
                # that are close to a known future anchor.
                for frame_i in range(gap_end, gap_start - 1, -1):
                    t = (frame_i - left_i) / (right_i - left_i)
                    prediction = xyz_out[left_i, marker_i] * (1.0 - t) + xyz_out[right_i, marker_i] * t
                    scored = []
                    for raw_label in unlabeled:
                        raw_i = label_to_i[raw_label]
                        if not valid_out[frame_i, raw_i]:
                            continue
                        pred_dist, support, mean_error = local_model_score(
                            marker,
                            xyz_out[frame_i, raw_i],
                            prediction,
                            xyz_out,
                            valid_out,
                            frame_i,
                            label_to_i,
                            refs,
                            neighbors,
                        )
                        if support >= 3 and pred_dist <= 70.0 and mean_error <= 30.0:
                            score = pred_dist + mean_error * 1.5 - support * 4.0
                            scored.append((score, raw_label, pred_dist, support, mean_error))
                    scored.sort(key=lambda item: item[0])
                    if not scored:
                        if candidates:
                            break
                        continue
                    best = scored[0]
                    margin = scored[1][0] - best[0] if len(scored) > 1 else 999.0
                    if margin < 40.0:
                        rejected.append(
                            {
                                "iteration": iteration,
                                "frame": frame_i + layout.first_frame,
                                "marker": marker,
                                "raw": best[1],
                                "reason": "ambiguous_candidate",
                                "margin": margin,
                            }
                        )
                        if candidates:
                            break
                        continue
                    candidates.append(
                        {
                            "iteration": iteration,
                            "frame": frame_i + layout.first_frame,
                            "marker": marker,
                            "raw": best[1],
                            "gap_start": gap_start + layout.first_frame,
                            "gap_end": gap_end + layout.first_frame,
                            "left_frame": left_i + layout.first_frame,
                            "right_frame": right_i + layout.first_frame,
                            "score": best[0],
                            "pred_dist": best[2],
                            "support": best[3],
                            "seg_error": best[4],
                            "margin": margin,
                            "method": "iterative_flicker_grey_connect",
                        }
                    )
                    if len(candidates) >= args.max_frames_per_gap_pass:
                        break

                for item in candidates:
                    frame_i = int(item["frame"]) - layout.first_frame
                    raw_i = label_to_i[item["raw"]]
                    if valid_out[frame_i, marker_i] or not valid_out[frame_i, raw_i]:
                        rejected.append({**item, "reason": "point_state_changed"})
                        continue
                    xyz_out[frame_i, marker_i] = xyz_out[frame_i, raw_i]
                    residual_out[frame_i, marker_i] = residual_out[frame_i, raw_i]
                    xyz_out[frame_i, raw_i] = 0.0
                    residual_out[frame_i, raw_i] = -1.0
                    valid_out[frame_i, marker_i] = True
                    valid_out[frame_i, raw_i] = False
                    accepted.append(item)
                    iteration_rows.append(item)

        if not iteration_rows:
            break

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "accepted_flicker_grey_iterative.csv", accepted)
    base.write_csv(args.report_dir / "rejected_flicker_grey_iterative.csv", rejected)

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
    base.write_csv(args.report_dir / "missing_summary_after_flicker_grey.csv", summary)
    print(f"accepted={len(accepted)}")
    print(f"by_marker={dict(Counter(row['marker'] for row in accepted))}")
    print(f"rejected={len(rejected)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
