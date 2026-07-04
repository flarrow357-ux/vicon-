from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import conservative_c3d_connect as base


def ranges(valid_col: np.ndarray, start_i: int, end_i: int):
    out = []
    i = start_i
    while i <= end_i:
        if valid_col[i]:
            i += 1
            continue
        gap_start = i
        while i <= end_i and not valid_col[i]:
            i += 1
        out.append((gap_start, i - 1))
    return out


def nearest_bounds(valid_col: np.ndarray, gap_start: int, gap_end: int, start_i: int, end_i: int):
    left = None
    right = None
    for frame_i in range(gap_start - 1, start_i - 1, -1):
        if valid_col[frame_i]:
            left = frame_i
            break
    for frame_i in range(gap_end + 1, end_i + 1):
        if valid_col[frame_i]:
            right = frame_i
            break
    return left, right


def interpolate(xyz: np.ndarray, residual: np.ndarray, marker_i: int, frame_i: int, left_i: int, right_i: int):
    t = (frame_i - left_i) / (right_i - left_i)
    pos = xyz[left_i, marker_i] * (1.0 - t) + xyz[right_i, marker_i] * t
    res = residual[left_i, marker_i] * (1.0 - t) + residual[right_i, marker_i] * t
    return pos, float(res)


def model_score(
    marker: str,
    pos: np.ndarray,
    pred: np.ndarray,
    xyz: np.ndarray,
    valid: np.ndarray,
    frame_i: int,
    label_to_i: dict[str, int],
    refs: dict[tuple[str, str], float],
    neighbors: dict[str, list[str]],
):
    pred_dist = float(np.linalg.norm(pos - pred))
    support = 0
    errors = []
    for neighbor in neighbors.get(marker, []):
        neighbor_i = label_to_i[neighbor]
        if not valid[frame_i, neighbor_i]:
            continue
        ref = refs.get((marker, neighbor))
        if ref is None:
            continue
        error = abs(float(np.linalg.norm(pos - xyz[frame_i, neighbor_i])) - ref)
        errors.append(error)
        tolerance = max(50.0, min(110.0, ref * 0.38))
        if error <= tolerance:
            support += 1
    mean_error = float(np.mean(errors)) if errors else 999.0
    return pred_dist, support, mean_error


def risk_level(gap_len: int, method: str, mean_error: float | None):
    if method == "grey_connect":
        if gap_len <= 10 and (mean_error is None or mean_error <= 25.0):
            return "low"
        if gap_len <= 80:
            return "medium"
        return "high"
    if gap_len <= 2:
        return "low"
    if gap_len <= 10:
        return "medium"
    return "high"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
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
    unlabeled = [label for label in layout.labels if label.startswith("*")]

    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    grey_rows = []
    fill_rows = []
    failed = []

    for marker in markers:
        marker_i = label_to_i[marker]
        for gap_start, gap_end in ranges(valid_out[:, marker_i], start_i, end_i):
            left_i, right_i = nearest_bounds(valid_out[:, marker_i], gap_start, gap_end, start_i, end_i)
            gap_len = gap_end - gap_start + 1
            if left_i is None or right_i is None:
                failed.append(
                    {
                        "marker": marker,
                        "gap_start": gap_start + layout.first_frame,
                        "gap_end": gap_end + layout.first_frame,
                        "gap_len": gap_len,
                        "reason": "no_bounds",
                    }
                )
                continue

            for frame_i in range(gap_start, gap_end + 1):
                pred, pred_res = interpolate(xyz_out, residual_out, marker_i, frame_i, left_i, right_i)
                candidates = []
                for raw_label in unlabeled:
                    raw_i = label_to_i[raw_label]
                    if not valid_out[frame_i, raw_i]:
                        continue
                    pred_dist, support, mean_error = model_score(
                        marker,
                        xyz_out[frame_i, raw_i],
                        pred,
                        xyz_out,
                        valid_out,
                        frame_i,
                        label_to_i,
                        refs,
                        neighbors,
                    )
                    required = base.support_requirement(marker)
                    if support >= required and (
                        (support >= 3 and pred_dist <= 95.0 and mean_error <= 45.0)
                        or (support >= 2 and pred_dist <= 55.0 and mean_error <= 30.0)
                    ):
                        score = pred_dist + mean_error * 1.2 - support * 5.0
                        candidates.append((score, raw_label, pred_dist, support, mean_error))
                candidates.sort(key=lambda item: item[0])
                if candidates:
                    best = candidates[0]
                    margin = candidates[1][0] - best[0] if len(candidates) > 1 else 999.0
                    if margin >= 10.0:
                        raw_i = label_to_i[best[1]]
                        xyz_out[frame_i, marker_i] = xyz_out[frame_i, raw_i]
                        residual_out[frame_i, marker_i] = residual_out[frame_i, raw_i]
                        xyz_out[frame_i, raw_i] = 0.0
                        residual_out[frame_i, raw_i] = -1.0
                        valid_out[frame_i, marker_i] = True
                        valid_out[frame_i, raw_i] = False
                        grey_rows.append(
                            {
                                "frame": frame_i + layout.first_frame,
                                "marker": marker,
                                "raw": best[1],
                                "gap_start": gap_start + layout.first_frame,
                                "gap_end": gap_end + layout.first_frame,
                                "gap_len": gap_len,
                                "left_frame": left_i + layout.first_frame,
                                "right_frame": right_i + layout.first_frame,
                                "pred_dist": best[2],
                                "support": best[3],
                                "mean_error": best[4],
                                "margin": margin,
                                "risk": risk_level(gap_len, "grey_connect", best[4]),
                                "method": "bold_grey_connect",
                            }
                        )
                        continue

                xyz_out[frame_i, marker_i] = pred
                residual_out[frame_i, marker_i] = pred_res if pred_res != -1.0 else 0.0
                valid_out[frame_i, marker_i] = True
                fill_rows.append(
                    {
                        "frame": frame_i + layout.first_frame,
                        "marker": marker,
                        "gap_start": gap_start + layout.first_frame,
                        "gap_end": gap_end + layout.first_frame,
                        "gap_len": gap_len,
                        "left_frame": left_i + layout.first_frame,
                        "right_frame": right_i + layout.first_frame,
                        "risk": risk_level(gap_len, "interpolation", None),
                        "method": "bold_linear_interpolation",
                    }
                )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    base.write_points(data, layout, xyz_out, residual_out, args.output)
    base.write_csv(args.report_dir / "bold_grey_connections.csv", grey_rows)
    base.write_csv(args.report_dir / "bold_interpolated_fills.csv", fill_rows)
    base.write_csv(args.report_dir / "bold_failed.csv", failed)

    after_valid = base.is_valid(xyz_out, residual_out)
    summary = []
    for marker in markers:
        marker_i = label_to_i[marker]
        before_missing = int(np.sum(~valid[start_i : end_i + 1, marker_i]))
        after_missing = int(np.sum(~after_valid[start_i : end_i + 1, marker_i]))
        if before_missing or after_missing:
            summary.append(
                {
                    "marker": marker,
                    "before_missing": before_missing,
                    "after_missing": after_missing,
                    "resolved": before_missing - after_missing,
                }
            )
    base.write_csv(args.report_dir / "bold_missing_summary.csv", summary)

    print(f"grey_connected={len(grey_rows)}")
    print(f"interpolated={len(fill_rows)}")
    print(f"failed={len(failed)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
