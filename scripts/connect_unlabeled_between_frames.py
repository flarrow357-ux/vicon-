from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyCGM2.Tools import btkTools


@dataclass(frozen=True)
class SegmentModel:
    name: str
    labels: list[str]
    coords: np.ndarray


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
        if not line or "," in line:
            break
        names.append(line)
    return names


def parse_vst_segments(vst_file: Path, marker_names: list[str]) -> list[SegmentModel]:
    root = ET.parse(vst_file).getroot()
    pattern = re.compile(r"(.+)_\1(\d+)_(x|y|z)$")
    order: list[str] = []
    values: dict[str, dict[int, dict[str, float]]] = {}
    for param in root.findall(".//Parameter"):
        match = pattern.match(param.attrib.get("NAME", ""))
        if not match:
            continue
        segment, marker_index, axis = match.groups()
        if segment not in values:
            values[segment] = {}
            order.append(segment)
        values[segment].setdefault(int(marker_index), {})[axis] = float(param.attrib["VALUE"])

    cursor = 0
    segments: list[SegmentModel] = []
    for segment in order:
        coords = []
        for index in sorted(values[segment]):
            axes = values[segment][index]
            coords.append([axes["x"], axes["y"], axes["z"]])
        labels = marker_names[cursor : cursor + len(coords)]
        if len(labels) != len(coords):
            break
        cursor += len(coords)
        segments.append(SegmentModel(segment, labels, np.asarray(coords, dtype=float)))
    return segments


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


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_points(acq):
    labels = [acq.GetPoint(i).GetLabel() for i in range(acq.GetPointNumber())]
    values = {}
    valid = {}
    residuals = {}
    for index, label in enumerate(labels):
        point = acq.GetPoint(index)
        values[label] = point.GetValues().astype(float)
        residuals[label] = point.GetResiduals().copy()
        valid[label] = residuals[label][:, 0] != -1.0
    return labels, values, valid, residuals


def get_gap_ranges(ok: np.ndarray, start_index: int, end_index: int):
    gaps = []
    i = start_index
    while i <= end_index:
        if ok[i]:
            i += 1
            continue
        start = i
        while i <= end_index and not ok[i]:
            i += 1
        gaps.append((start, i - 1))
    return gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--vst", required=True, type=Path)
    parser.add_argument("--mkr", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--distance-threshold-mm", type=float, default=45.0)
    parser.add_argument("--ambiguity-margin-mm", type=float, default=20.0)
    parser.add_argument("--fit-threshold-mm", type=float, default=35.0)
    parser.add_argument("--boundary-threshold-mm", type=float, default=25.0)
    parser.add_argument("--min-boundary-coverage", type=float, default=0.35)
    parser.add_argument("--step-threshold-mm", type=float, default=80.0)
    parser.add_argument("--step-ambiguity-margin-mm", type=float, default=15.0)
    parser.add_argument("--write-c3d", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    marker_names = parse_marker_names(args.mkr)
    segments = parse_vst_segments(args.vst, marker_names)
    segment_by_marker = {
        label: segment for segment in segments for label in segment.labels
    }
    model_index = {
        label: idx for segment in segments for idx, label in enumerate(segment.labels)
    }

    acq = btkTools.smartReader(str(args.c3d))
    first_frame = acq.GetFirstFrame()
    frame_count = acq.GetPointFrameNumber()
    start_index = args.start_frame - first_frame
    end_index = args.end_frame - first_frame
    if start_index < 0 or end_index >= frame_count:
        raise ValueError(
            f"帧范围 {args.start_frame}-{args.end_frame} 超出 C3D 范围 "
            f"{first_frame}-{acq.GetLastFrame()}"
        )

    labels, values, valid, residuals = load_points(acq)
    unlabeled = [label for label in labels if label.startswith("*")]
    used_unlabeled_frames: dict[str, set[int]] = {label: set() for label in unlabeled}

    connection_rows = []
    uncertain_rows = []
    propagation_rows = []

    def propagate(direction: str):
        if direction == "forward":
            frame_iter = range(start_index + 1, end_index + 1)
            previous = lambda frame: frame - 1
            previous2 = lambda frame: frame - 2
        else:
            frame_iter = range(end_index - 1, start_index - 1, -1)
            previous = lambda frame: frame + 1
            previous2 = lambda frame: frame + 2

        for frame in frame_iter:
            raw_available = [
                raw_label for raw_label in unlabeled if valid[raw_label][frame]
            ]
            if not raw_available:
                continue
            candidates = []
            for marker in marker_names:
                if valid[marker][frame]:
                    continue
                prev = previous(frame)
                prev2 = previous2(frame)
                if prev < 0 or prev >= frame_count or not valid[marker][prev]:
                    continue
                predicted = values[marker][prev].copy()
                if 0 <= prev2 < frame_count and valid[marker][prev2]:
                    predicted = values[marker][prev] + (values[marker][prev] - values[marker][prev2])
                distances = sorted(
                    (
                        float(np.linalg.norm(values[raw_label][frame] - predicted)),
                        raw_label,
                    )
                    for raw_label in raw_available
                )
                if not distances:
                    continue
                best_distance, best_raw = distances[0]
                second_distance = distances[1][0] if len(distances) > 1 else None
                ambiguous = (
                    second_distance is not None
                    and second_distance - best_distance < args.step_ambiguity_margin_mm
                )
                if best_distance <= args.step_threshold_mm and not ambiguous:
                    candidates.append((best_distance, marker, best_raw, second_distance))
                elif best_distance <= args.step_threshold_mm:
                    uncertain_rows.append(
                        {
                            "frame": frame + first_frame,
                            "marker": marker,
                            "segment": segment_by_marker[marker].name,
                            "reason": f"ambiguous_{direction}_step",
                            "best_unlabeled": best_raw,
                            "best_distance_mm": round(best_distance, 3),
                            "second_distance_mm": round(second_distance, 3)
                            if second_distance is not None
                            else "",
                            "fit_error_mm": "",
                        }
                    )

            used_markers = set()
            used_raw = set()
            for distance, marker, raw_label, second_distance in sorted(candidates):
                if marker in used_markers or raw_label in used_raw:
                    continue
                if not valid[raw_label][frame] or valid[marker][frame]:
                    continue
                values[marker][frame] = values[raw_label][frame]
                valid[marker][frame] = True
                residuals[marker][frame, 0] = 0.0
                residuals[raw_label][frame, 0] = -1.0
                valid[raw_label][frame] = False
                used_unlabeled_frames[raw_label].add(frame)
                used_markers.add(marker)
                used_raw.add(raw_label)
                propagation_rows.append(
                    {
                        "frame": frame + first_frame,
                        "marker": marker,
                        "source_unlabeled": raw_label,
                        "distance_mm": round(distance, 3),
                        "second_distance_mm": round(second_distance, 3)
                        if second_distance is not None
                        else "",
                        "direction": direction,
                    }
                )

    propagate("forward")
    propagate("backward")

    for frame in range(start_index, end_index + 1):
        assigned_unlabeled_this_frame: set[str] = set()
        changed = True
        pass_count = 0
        while changed and pass_count < 3:
            changed = False
            pass_count += 1
            for marker in marker_names:
                if valid[marker][frame]:
                    continue
                segment = segment_by_marker[marker]
                present_indexes = [
                    idx
                    for idx, label in enumerate(segment.labels)
                    if label != marker and valid[label][frame]
                ]
                if len(present_indexes) < 3:
                    continue
                model_obs = segment.coords[present_indexes]
                data_obs = np.asarray(
                    [values[segment.labels[idx]][frame] for idx in present_indexes],
                    dtype=float,
                )
                rotation, translation = rigid_transform(model_obs, data_obs)
                reconstructed = model_obs @ rotation.T + translation
                fit_error = float(
                    np.sqrt(np.mean(np.sum((reconstructed - data_obs) ** 2, axis=1)))
                )
                if fit_error > args.fit_threshold_mm:
                    continue
                predicted = segment.coords[model_index[marker]] @ rotation.T + translation

                candidates = []
                for raw_label in unlabeled:
                    if raw_label in assigned_unlabeled_this_frame:
                        continue
                    if frame in used_unlabeled_frames[raw_label]:
                        continue
                    if not valid[raw_label][frame]:
                        continue
                    distance = float(np.linalg.norm(values[raw_label][frame] - predicted))
                    candidates.append((distance, raw_label))
                candidates.sort(key=lambda item: item[0])
                if not candidates:
                    continue
                best_distance, best_label = candidates[0]
                second_distance = candidates[1][0] if len(candidates) > 1 else None
                ambiguous = (
                    second_distance is not None
                    and second_distance - best_distance < args.ambiguity_margin_mm
                )
                absolute_frame = frame + first_frame
                if best_distance <= args.distance_threshold_mm and not ambiguous:
                    values[marker][frame] = values[best_label][frame]
                    valid[marker][frame] = True
                    residuals[marker][frame, 0] = 0.0
                    residuals[best_label][frame, 0] = -1.0
                    valid[best_label][frame] = False
                    used_unlabeled_frames[best_label].add(frame)
                    assigned_unlabeled_this_frame.add(best_label)
                    changed = True
                    connection_rows.append(
                        {
                            "frame": absolute_frame,
                            "marker": marker,
                            "segment": segment.name,
                            "source_unlabeled": best_label,
                            "distance_mm": round(best_distance, 3),
                            "second_distance_mm": round(second_distance, 3)
                            if second_distance is not None
                            else "",
                            "fit_error_mm": round(fit_error, 3),
                            "pass": pass_count,
                        }
                    )
                elif best_distance <= args.distance_threshold_mm:
                    uncertain_rows.append(
                        {
                            "frame": absolute_frame,
                            "marker": marker,
                            "segment": segment.name,
                            "reason": "ambiguous_nearest_unlabeled",
                            "best_unlabeled": best_label,
                            "best_distance_mm": round(best_distance, 3),
                            "second_distance_mm": round(second_distance, 3)
                            if second_distance is not None
                            else "",
                            "fit_error_mm": round(fit_error, 3),
                        }
                    )

    boundary_rows = []
    for marker in marker_names:
        for gap_start, gap_end in get_gap_ranges(valid[marker], start_index, end_index):
            gap_len = gap_end - gap_start + 1
            left_frame = gap_start - 1 if gap_start - 1 >= 0 and valid[marker][gap_start - 1] else None
            right_frame = (
                gap_end + 1
                if gap_end + 1 < frame_count and valid[marker][gap_end + 1]
                else None
            )
            if left_frame is None and right_frame is None:
                continue
            candidates = []
            for raw_label in unlabeled:
                frames = [
                    frame
                    for frame in range(gap_start, gap_end + 1)
                    if valid[raw_label][frame] and frame not in used_unlabeled_frames[raw_label]
                ]
                if not frames:
                    continue
                coverage = len(frames) / gap_len
                if coverage < args.min_boundary_coverage:
                    continue
                first_raw = frames[0]
                last_raw = frames[-1]
                distances = []
                if left_frame is not None:
                    distances.append(
                        float(np.linalg.norm(values[raw_label][first_raw] - values[marker][left_frame]))
                    )
                if right_frame is not None:
                    distances.append(
                        float(np.linalg.norm(values[raw_label][last_raw] - values[marker][right_frame]))
                    )
                if not distances:
                    continue
                max_distance = max(distances)
                mean_distance = sum(distances) / len(distances)
                candidates.append((max_distance, mean_distance, raw_label, frames, coverage))
            candidates.sort(key=lambda item: (item[0], item[1], -len(item[3])))
            if not candidates:
                continue
            best = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            ambiguous = (
                second is not None
                and second[0] - best[0] < args.ambiguity_margin_mm
            )
            if best[0] <= args.boundary_threshold_mm and not ambiguous:
                _, mean_distance, raw_label, frames, coverage = best
                for frame in frames:
                    values[marker][frame] = values[raw_label][frame]
                    valid[marker][frame] = True
                    residuals[marker][frame, 0] = 0.0
                    residuals[raw_label][frame, 0] = -1.0
                    valid[raw_label][frame] = False
                    used_unlabeled_frames[raw_label].add(frame)
                boundary_rows.append(
                    {
                        "marker": marker,
                        "gap_start_frame": gap_start + first_frame,
                        "gap_end_frame": gap_end + first_frame,
                        "gap_length_frames": gap_len,
                        "source_unlabeled": raw_label,
                        "connected_frames": len(frames),
                        "coverage": round(coverage, 4),
                        "max_boundary_distance_mm": round(best[0], 3),
                        "mean_boundary_distance_mm": round(mean_distance, 3),
                        "second_max_boundary_distance_mm": round(second[0], 3)
                        if second is not None
                        else "",
                    }
                )
            elif best[0] <= args.boundary_threshold_mm:
                uncertain_rows.append(
                    {
                        "frame": gap_start + first_frame,
                        "marker": marker,
                        "segment": segment_by_marker[marker].name,
                        "reason": "ambiguous_boundary_unlabeled",
                        "best_unlabeled": best[2],
                        "best_distance_mm": round(best[0], 3),
                        "second_distance_mm": round(second[0], 3)
                        if second is not None
                        else "",
                        "fit_error_mm": "",
                    }
                )

    marker_rows = []
    for marker in marker_names:
        roi_ok = valid[marker][start_index : end_index + 1]
        original_point = acq.GetPoint(marker)
        before_ok = original_point.GetResiduals()[start_index : end_index + 1, 0] != -1.0
        marker_rows.append(
            {
                "marker": marker,
                "observed_before": int(before_ok.sum()),
                "connected_from_unlabeled": int(roi_ok.sum() - before_ok.sum()),
                "missing_after": int((~roi_ok).sum()),
                "visible_after": int(roi_ok.sum()),
            }
        )

    gap_rows = []
    for marker in marker_names:
        for gap_start, gap_end in get_gap_ranges(valid[marker], start_index, end_index):
            gap_rows.append(
                {
                    "marker": marker,
                    "gap_start_frame": gap_start + first_frame,
                    "gap_end_frame": gap_end + first_frame,
                    "gap_length_frames": gap_end - gap_start + 1,
                }
            )

    write_csv(args.out_dir / "connected_points.csv", connection_rows)
    write_csv(args.out_dir / "propagation_connected_points.csv", propagation_rows)
    write_csv(args.out_dir / "boundary_connected_gaps.csv", boundary_rows)
    write_csv(args.out_dir / "uncertain_points.csv", uncertain_rows)
    write_csv(args.out_dir / "marker_connection_summary.csv", marker_rows)
    write_csv(args.out_dir / "remaining_gaps.csv", gap_rows)

    output_c3d = ""
    if args.write_c3d:
        output_c3d = str(args.out_dir / f"{args.c3d.stem}_connected_{args.start_frame}_{args.end_frame}.c3d")
        for label in marker_names + unlabeled:
            point = acq.GetPoint(label)
            point.SetValues(values[label])
            point.SetResiduals(residuals[label])
        btkTools.smartWriter(acq, output_c3d)

    summary = {
        "input_c3d": str(args.c3d),
        "output_c3d": output_c3d,
        "first_frame": first_frame,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "markers": len(marker_names),
        "unlabeled_tracks": len(unlabeled),
        "connected_points": len(connection_rows),
        "propagation_connected_points": len(propagation_rows),
        "boundary_connected_gaps": len(boundary_rows),
        "boundary_connected_points": int(sum(row["connected_frames"] for row in boundary_rows)),
        "uncertain_points": len(uncertain_rows),
        "remaining_gap_segments": len(gap_rows),
        "distance_threshold_mm": args.distance_threshold_mm,
        "ambiguity_margin_mm": args.ambiguity_margin_mm,
        "fit_threshold_mm": args.fit_threshold_mm,
        "boundary_threshold_mm": args.boundary_threshold_mm,
        "min_boundary_coverage": args.min_boundary_coverage,
        "step_threshold_mm": args.step_threshold_mm,
        "step_ambiguity_margin_mm": args.step_ambiguity_margin_mm,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
