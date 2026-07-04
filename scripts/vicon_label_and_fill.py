from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyCGM2.Tools import btkTools


COMBO_CACHE: dict[tuple[int, int], np.ndarray] = {}


@dataclass(frozen=True)
class SegmentModel:
    name: str
    labels: list[str]
    coords: np.ndarray


@dataclass(frozen=True)
class Assignment:
    segment: str
    labels: list[str]
    tracks: tuple[int, ...]
    score_mm: float
    max_error_mm: float
    coverage: float
    pair_stability_mm: float


def parse_marker_names(marker_file: Path) -> list[str]:
    names: list[str] = []
    in_display = False
    for raw_line in marker_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line == "[Display]":
            in_display = True
            continue
        if not in_display:
            continue
        if not line:
            break
        if "," in line:
            break
        names.append(line)
    return names


def parse_vst_segments(vst_file: Path, marker_names: list[str]) -> list[SegmentModel]:
    root = ET.parse(vst_file).getroot()
    segment_order: list[str] = []
    values: dict[str, dict[int, dict[str, float]]] = {}
    pattern = re.compile(r"(.+)_\1(\d+)_(x|y|z)$")

    for param in root.findall(".//Parameter"):
        name = param.attrib.get("NAME", "")
        match = pattern.match(name)
        if not match:
            continue
        segment, marker_index, axis = match.groups()
        if segment not in values:
            values[segment] = {}
            segment_order.append(segment)
        values[segment].setdefault(int(marker_index), {})[axis] = float(
            param.attrib["VALUE"]
        )

    segments: list[SegmentModel] = []
    cursor = 0
    for segment in segment_order:
        indexes = sorted(values[segment])
        coords = []
        for index in indexes:
            axes = values[segment][index]
            coords.append([axes["x"], axes["y"], axes["z"]])
        count = len(coords)
        labels = marker_names[cursor : cursor + count]
        if len(labels) != count:
            # Some VST files include tool/object segments that are not listed in
            # the MKR display set. Ignore those unless their marker names are
            # explicitly available.
            break
        cursor += count
        segments.append(SegmentModel(segment, labels, np.asarray(coords, dtype=float)))

    if cursor != len(marker_names):
        raise ValueError(
            f"VST 解析到 {cursor} 个 marker，但 MKR 中有 {len(marker_names)} 个 marker"
        )
    return segments


def load_tracks(c3d_file: Path):
    acq = btkTools.smartReader(str(c3d_file))
    point_count = acq.GetPointNumber()
    frame_count = acq.GetPointFrameNumber()
    values = np.empty((point_count, frame_count, 3), dtype=float)
    valid = np.empty((point_count, frame_count), dtype=bool)
    labels: list[str] = []
    residuals = []

    for point_index in range(point_count):
        point = acq.GetPoint(point_index)
        labels.append(point.GetLabel())
        values[point_index] = point.GetValues()
        residual = point.GetResiduals()
        residuals.append(residual.copy())
        valid[point_index] = residual[:, 0] != -1.0

    return acq, labels, values, valid, residuals


def compute_pair_stats(values: np.ndarray, valid: np.ndarray):
    track_count = values.shape[0]
    median = np.full((track_count, track_count), np.nan, dtype=float)
    mad = np.full((track_count, track_count), np.nan, dtype=float)
    coverage = np.zeros((track_count, track_count), dtype=float)

    for i in range(track_count):
        median[i, i] = 0.0
        mad[i, i] = 0.0
        coverage[i, i] = valid[i].mean()
        for j in range(i + 1, track_count):
            ok = valid[i] & valid[j]
            coverage[i, j] = coverage[j, i] = ok.mean()
            if ok.sum() < 50:
                continue
            distances = np.linalg.norm(values[i, ok] - values[j, ok], axis=1)
            med = float(np.median(distances))
            robust_mad = float(np.median(np.abs(distances - med)))
            median[i, j] = median[j, i] = med
            mad[i, j] = mad[j, i] = robust_mad
    return median, mad, coverage


def model_distance_matrix(coords: np.ndarray) -> np.ndarray:
    count = coords.shape[0]
    out = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            out[i, j] = out[j, i] = np.linalg.norm(coords[i] - coords[j])
    return out


def assignment_score(
    model_dist: np.ndarray,
    track_order: tuple[int, ...],
    pair_median: np.ndarray,
    pair_mad: np.ndarray,
    pair_coverage: np.ndarray,
):
    errors = []
    stabilities = []
    coverages = []
    for a, b in itertools.combinations(range(len(track_order)), 2):
        i = track_order[a]
        j = track_order[b]
        observed = pair_median[i, j]
        if np.isnan(observed):
            return math.inf, math.inf, 0.0, math.inf
        errors.append(abs(observed - model_dist[a, b]))
        stabilities.append(pair_mad[i, j])
        coverages.append(pair_coverage[i, j])

    errors_np = np.asarray(errors, dtype=float)
    stability = float(np.nanmedian(stabilities))
    coverage = float(np.nanmedian(coverages))
    rms = float(np.sqrt(np.mean(errors_np**2)))
    score = rms + 0.35 * stability + 15.0 * max(0.0, 0.70 - coverage)
    return score, float(errors_np.max()), coverage, stability


def get_combos(track_count: int, count: int) -> np.ndarray:
    key = (track_count, count)
    if key not in COMBO_CACHE:
        COMBO_CACHE[key] = np.asarray(
            list(itertools.combinations(range(track_count), count)), dtype=np.int16
        )
    return COMBO_CACHE[key]


def find_segment_candidates(
    segment: SegmentModel,
    pair_median: np.ndarray,
    pair_mad: np.ndarray,
    pair_coverage: np.ndarray,
    top_n: int,
    coarse_limit_mm: float,
    max_refine: int,
) -> list[Assignment]:
    model_dist = model_distance_matrix(segment.coords)
    upper = np.triu_indices(len(segment.labels), 1)
    model_sorted = np.sort(model_dist[upper])
    track_count = pair_median.shape[0]
    count = len(segment.labels)
    candidates: list[Assignment] = []
    combos = get_combos(track_count, count)
    pair_positions = list(itertools.combinations(range(count), 2))

    distance_columns = []
    coverage_columns = []
    stability_columns = []
    for a, b in pair_positions:
        left = combos[:, a]
        right = combos[:, b]
        distance_columns.append(pair_median[left, right])
        coverage_columns.append(pair_coverage[left, right])
        stability_columns.append(pair_mad[left, right])

    combo_distances = np.column_stack(distance_columns)
    combo_coverages = np.column_stack(coverage_columns)
    combo_stabilities = np.column_stack(stability_columns)
    finite_mask = np.isfinite(combo_distances).all(axis=1)
    finite_mask &= np.nanmedian(combo_coverages, axis=1) >= 0.45
    finite_mask &= np.nanmedian(combo_stabilities, axis=1) <= coarse_limit_mm

    if not finite_mask.any():
        return []

    observed_sorted = np.sort(combo_distances[finite_mask], axis=1)
    coarse_rms = np.sqrt(np.mean((observed_sorted - model_sorted) ** 2, axis=1))
    finite_indexes = np.flatnonzero(finite_mask)
    keep = np.flatnonzero(coarse_rms <= coarse_limit_mm)
    if keep.size == 0:
        return []
    keep = keep[np.argsort(coarse_rms[keep])[:max_refine]]

    for combo_index in finite_indexes[keep]:
        combo = tuple(int(value) for value in combos[combo_index])
        best_for_combo = None
        for perm in itertools.permutations(combo):
            score, max_error, coverage, stability = assignment_score(
                model_dist, perm, pair_median, pair_mad, pair_coverage
            )
            if best_for_combo is None or score < best_for_combo.score_mm:
                best_for_combo = Assignment(
                    segment.name,
                    segment.labels,
                    tuple(perm),
                    score,
                    max_error,
                    coverage,
                    stability,
                )
        if best_for_combo is not None and math.isfinite(best_for_combo.score_mm):
            candidates.append(best_for_combo)

    candidates.sort(key=lambda item: item.score_mm)
    return candidates[:top_n]


def choose_unique_assignments(
    candidates_by_segment: dict[str, list[Assignment]],
    score_limit_mm: float,
    max_error_limit_mm: float,
) -> list[Assignment]:
    # Small exhaustive search over the top few candidates per segment gives better
    # conflict handling than a greedy pass while staying transparent.
    ordered_segments = sorted(
        candidates_by_segment,
        key=lambda name: candidates_by_segment[name][0].score_mm
        if candidates_by_segment[name]
        else math.inf,
    )
    best: tuple[float, list[Assignment]] = (math.inf, [])

    def backtrack(index: int, used: set[int], chosen: list[Assignment], total: float):
        nonlocal best
        if total >= best[0]:
            return
        if index == len(ordered_segments):
            best = (total, chosen.copy())
            return

        segment = ordered_segments[index]
        accepted_any = False
        for candidate in candidates_by_segment[segment]:
            if candidate.score_mm > score_limit_mm:
                continue
            if candidate.max_error_mm > max_error_limit_mm:
                continue
            if any(track in used for track in candidate.tracks):
                continue
            accepted_any = True
            chosen.append(candidate)
            backtrack(index + 1, used | set(candidate.tracks), chosen, total + candidate.score_mm)
            chosen.pop()

        # Allow a segment to stay unresolved rather than forcing a bad match.
        penalty = score_limit_mm * 2.5
        backtrack(index + 1, used, chosen, total + penalty)
        if not accepted_any:
            return

    backtrack(0, set(), [], 0.0)
    return best[1]


def rigid_transform(model_points: np.ndarray, observed_points: np.ndarray):
    model_centroid = model_points.mean(axis=0)
    observed_centroid = observed_points.mean(axis=0)
    model_centered = model_points - model_centroid
    observed_centered = observed_points - observed_centroid
    h = model_centered.T @ observed_centered
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = observed_centroid - model_centroid @ rotation.T
    return rotation, translation


def apply_rigid_fill(
    segment: SegmentModel,
    assignment: Assignment,
    values: np.ndarray,
    valid: np.ndarray,
    fit_limit_mm: float,
):
    count = len(segment.labels)
    frame_count = values.shape[1]
    output = np.full((count, frame_count, 3), np.nan, dtype=float)
    output_valid = np.zeros((count, frame_count), dtype=bool)
    source = np.full((count, frame_count), "missing", dtype=object)
    fit_error = np.full(frame_count, np.nan, dtype=float)

    for model_idx, track_idx in enumerate(assignment.tracks):
        ok = valid[track_idx]
        output[model_idx, ok] = values[track_idx, ok]
        output_valid[model_idx, ok] = True
        source[model_idx, ok] = "observed"

    for frame in range(frame_count):
        observed_indexes = [
            idx for idx, track_idx in enumerate(assignment.tracks) if valid[track_idx, frame]
        ]
        if len(observed_indexes) < 3:
            continue
        model_obs = segment.coords[observed_indexes]
        data_obs = np.asarray(
            [values[assignment.tracks[idx], frame] for idx in observed_indexes],
            dtype=float,
        )
        rotation, translation = rigid_transform(model_obs, data_obs)
        reconstructed_obs = model_obs @ rotation.T + translation
        frame_error = float(np.sqrt(np.mean(np.sum((reconstructed_obs - data_obs) ** 2, axis=1))))
        fit_error[frame] = frame_error
        if frame_error > fit_limit_mm:
            continue
        all_reconstructed = segment.coords @ rotation.T + translation
        for idx in range(count):
            if not output_valid[idx, frame]:
                output[idx, frame] = all_reconstructed[idx]
                output_valid[idx, frame] = True
                source[idx, frame] = "rigid_fill"

    return output, output_valid, source, fit_error


def interpolate_short_gaps(
    marker_values: np.ndarray,
    marker_valid: np.ndarray,
    source: np.ndarray,
    max_gap_frames: int,
):
    valid = marker_valid.copy()
    values = marker_values.copy()
    frame_count = len(valid)
    filled = 0
    cursor = 0
    while cursor < frame_count:
        if valid[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor < frame_count and not valid[cursor]:
            cursor += 1
        end = cursor - 1
        gap_len = end - start + 1
        left = start - 1
        right = end + 1
        if (
            gap_len <= max_gap_frames
            and left >= 0
            and right < frame_count
            and valid[left]
            and valid[right]
        ):
            for frame in range(start, right):
                alpha = (frame - left) / (right - left)
                values[frame] = (1.0 - alpha) * values[left] + alpha * values[right]
                valid[frame] = True
                source[frame] = "short_interp"
                filled += 1
    return values, valid, source, filled


def write_outputs(
    acq,
    c3d_out: Path,
    assignments: list[Assignment],
    segment_by_name: dict[str, SegmentModel],
    values: np.ndarray,
    valid: np.ndarray,
    short_gap_frames: int,
    fit_limit_mm: float,
):
    from pyCGM2 import btk

    report_rows = []
    used_tracks = set()
    for assignment in assignments:
        segment = segment_by_name[assignment.segment]
        seg_values, seg_valid, seg_source, fit_error = apply_rigid_fill(
            segment, assignment, values, valid, fit_limit_mm=fit_limit_mm
        )
        for idx, label in enumerate(segment.labels):
            marker_values = seg_values[idx]
            marker_valid = seg_valid[idx]
            marker_source = seg_source[idx]
            before_observed = int((marker_source == "observed").sum())
            rigid_filled = int((marker_source == "rigid_fill").sum())
            marker_values, marker_valid, marker_source, short_filled = interpolate_short_gaps(
                marker_values,
                marker_valid,
                marker_source,
                short_gap_frames,
            )
            final_values = np.nan_to_num(marker_values)
            residuals = np.where(marker_valid, 0.0, -1.0).reshape((-1, 1))

            point = acq.GetPoint(assignment.tracks[idx])
            point.SetLabel(label)
            point.SetType(btk.btkPoint.Marker)
            point.SetValues(final_values)
            point.SetResiduals(residuals)
            used_tracks.add(assignment.tracks[idx])

            report_rows.append(
                {
                    "segment": segment.name,
                    "marker": label,
                    "source_track": f"*{assignment.tracks[idx]}",
                    "observed_frames": before_observed,
                    "rigid_filled_frames": rigid_filled,
                    "short_interp_frames": short_filled,
                    "remaining_missing_frames": int((~marker_valid).sum()),
                    "segment_score_mm": round(assignment.score_mm, 3),
                    "segment_max_pair_error_mm": round(assignment.max_error_mm, 3),
                    "median_rigid_fit_error_mm": round(float(np.nanmedian(fit_error)), 3)
                    if np.isfinite(fit_error).any()
                    else "",
                }
            )

    btkTools.smartWriter(acq, str(c3d_out))
    return report_rows, sorted(used_tracks)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--vst", required=True, type=Path)
    parser.add_argument("--mkr", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--write-c3d", action="store_true")
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--score-limit-mm", type=float, default=45.0)
    parser.add_argument("--max-error-limit-mm", type=float, default=70.0)
    parser.add_argument("--coarse-limit-mm", type=float, default=90.0)
    parser.add_argument("--max-refine", type=int, default=500)
    parser.add_argument("--rigid-fit-limit-mm", type=float, default=35.0)
    parser.add_argument("--short-gap-frames", type=int, default=6)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    marker_names = parse_marker_names(args.mkr)
    segments = parse_vst_segments(args.vst, marker_names)
    acq, track_labels, values, valid, _ = load_tracks(args.c3d)

    pair_median, pair_mad, pair_coverage = compute_pair_stats(values, valid)
    candidates_by_segment = {
        segment.name: find_segment_candidates(
            segment,
            pair_median,
            pair_mad,
            pair_coverage,
            top_n=args.top_candidates,
            coarse_limit_mm=args.coarse_limit_mm,
            max_refine=args.max_refine,
        )
        for segment in segments
    }
    assignments = choose_unique_assignments(
        candidates_by_segment,
        score_limit_mm=args.score_limit_mm,
        max_error_limit_mm=args.max_error_limit_mm,
    )

    candidate_rows = []
    for segment in segments:
        for rank, candidate in enumerate(candidates_by_segment[segment.name], start=1):
            candidate_rows.append(
                {
                    "segment": segment.name,
                    "rank": rank,
                    "labels": " ".join(candidate.labels),
                    "tracks": " ".join(f"*{track}" for track in candidate.tracks),
                    "score_mm": round(candidate.score_mm, 3),
                    "max_pair_error_mm": round(candidate.max_error_mm, 3),
                    "coverage": round(candidate.coverage, 4),
                    "pair_stability_mm": round(candidate.pair_stability_mm, 3),
                    "selected": candidate in assignments,
                }
            )
    write_csv(args.out_dir / "candidate_assignments.csv", candidate_rows)

    selected_rows = [
        {
            "segment": item.segment,
            "labels": " ".join(item.labels),
            "tracks": " ".join(f"*{track}" for track in item.tracks),
            "score_mm": round(item.score_mm, 3),
            "max_pair_error_mm": round(item.max_error_mm, 3),
            "coverage": round(item.coverage, 4),
            "pair_stability_mm": round(item.pair_stability_mm, 3),
        }
        for item in assignments
    ]
    write_csv(args.out_dir / "selected_assignments.csv", selected_rows)

    track_rows = []
    for idx, label in enumerate(track_labels):
        ok = valid[idx]
        track_rows.append(
            {
                "track": label,
                "frames_visible": int(ok.sum()),
                "first_visible_frame": int(np.argmax(ok) + acq.GetFirstFrame())
                if ok.any()
                else "",
                "last_visible_frame": int(len(ok) - 1 - np.argmax(ok[::-1]) + acq.GetFirstFrame())
                if ok.any()
                else "",
            }
        )
    write_csv(args.out_dir / "track_visibility.csv", track_rows)

    summary = {
        "input_c3d": str(args.c3d),
        "point_frames": acq.GetPointFrameNumber(),
        "point_frequency": acq.GetPointFrequency(),
        "model_marker_count": len(marker_names),
        "raw_track_count": len(track_labels),
        "selected_segment_count": len(assignments),
        "selected_marker_count": sum(len(item.labels) for item in assignments),
        "unresolved_segments": [
            segment.name for segment in segments if segment.name not in {a.segment for a in assignments}
        ],
        "thresholds": {
            "score_limit_mm": args.score_limit_mm,
            "max_error_limit_mm": args.max_error_limit_mm,
            "rigid_fit_limit_mm": args.rigid_fit_limit_mm,
            "short_gap_frames": args.short_gap_frames,
        },
    }

    if args.write_c3d:
        c3d_out = args.out_dir / f"{args.c3d.stem}_labeled_rigidfill.c3d"
        report_rows, used_tracks = write_outputs(
            acq,
            c3d_out,
            assignments,
            {segment.name: segment for segment in segments},
            values,
            valid,
            short_gap_frames=args.short_gap_frames,
            fit_limit_mm=args.rigid_fit_limit_mm,
        )
        write_csv(args.out_dir / "fill_report.csv", report_rows)
        summary["output_c3d"] = str(c3d_out)
        summary["used_raw_track_indices"] = used_tracks

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
