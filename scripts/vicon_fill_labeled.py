from __future__ import annotations

import argparse
import csv
import json
import math
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


def load_marker_arrays(acq, marker_names: list[str]):
    frame_count = acq.GetPointFrameNumber()
    values = {}
    valid = {}
    for label in marker_names:
        try:
            point = acq.GetPoint(label)
        except RuntimeError:
            values[label] = np.full((frame_count, 3), np.nan, dtype=float)
            valid[label] = np.zeros(frame_count, dtype=bool)
            continue
        residual = point.GetResiduals()[:, 0]
        ok = residual != -1.0
        data = point.GetValues().astype(float)
        data[~ok] = np.nan
        values[label] = data
        valid[label] = ok
    return values, valid


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


def segment_rigid_fill(
    segment: SegmentModel,
    values: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    fit_limit_mm: float,
):
    frame_count = next(iter(values.values())).shape[0]
    output = {label: values[label].copy() for label in segment.labels}
    output_valid = {label: valid[label].copy() for label in segment.labels}
    source = {
        label: np.where(valid[label], "observed", "missing").astype(object)
        for label in segment.labels
    }
    fit_errors = np.full(frame_count, np.nan, dtype=float)
    usable_frames = 0

    for frame in range(frame_count):
        present = [idx for idx, label in enumerate(segment.labels) if output_valid[label][frame]]
        if len(present) < 3:
            continue
        model_obs = segment.coords[present]
        data_obs = np.asarray([output[segment.labels[idx]][frame] for idx in present])
        rotation, translation = rigid_transform(model_obs, data_obs)
        reconstructed_obs = model_obs @ rotation.T + translation
        fit_error = float(np.sqrt(np.mean(np.sum((reconstructed_obs - data_obs) ** 2, axis=1))))
        fit_errors[frame] = fit_error
        if fit_error > fit_limit_mm:
            continue
        usable_frames += 1
        reconstructed_all = segment.coords @ rotation.T + translation
        for idx, label in enumerate(segment.labels):
            if not output_valid[label][frame]:
                output[label][frame] = reconstructed_all[idx]
                output_valid[label][frame] = True
                source[label][frame] = "rigid_fill"
    return output, output_valid, source, fit_errors, usable_frames


def interpolate_short_gaps(data, ok, source, max_gap_frames):
    data = data.copy()
    ok = ok.copy()
    source = source.copy()
    filled = 0
    i = 0
    while i < len(ok):
        if ok[i]:
            i += 1
            continue
        start = i
        while i < len(ok) and not ok[i]:
            i += 1
        end = i - 1
        gap_len = end - start + 1
        left = start - 1
        right = end + 1
        if (
            gap_len <= max_gap_frames
            and left >= 0
            and right < len(ok)
            and ok[left]
            and ok[right]
        ):
            for frame in range(start, right):
                alpha = (frame - left) / (right - left)
                data[frame] = (1 - alpha) * data[left] + alpha * data[right]
                ok[frame] = True
                source[frame] = "short_interp"
                filled += 1
    return data, ok, source, filled


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
    parser.add_argument("--rigid-fit-limit-mm", type=float, default=35.0)
    parser.add_argument("--short-gap-frames", type=int, default=6)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    marker_names = parse_marker_names(args.mkr)
    segments = parse_vst_segments(args.vst, marker_names)
    acq = btkTools.smartReader(str(args.c3d))
    values, valid = load_marker_arrays(acq, marker_names)

    segment_rows = []
    marker_rows = []
    final_values: dict[str, np.ndarray] = {}
    final_valid: dict[str, np.ndarray] = {}

    for segment in segments:
        seg_values, seg_valid, seg_source, fit_errors, usable_frames = segment_rigid_fill(
            segment, values, valid, args.rigid_fit_limit_mm
        )
        finite_errors = fit_errors[np.isfinite(fit_errors)]
        segment_rows.append(
            {
                "segment": segment.name,
                "markers": " ".join(segment.labels),
                "usable_rigid_frames": usable_frames,
                "median_fit_error_mm": round(float(np.median(finite_errors)), 3)
                if finite_errors.size
                else "",
                "p95_fit_error_mm": round(float(np.percentile(finite_errors, 95)), 3)
                if finite_errors.size
                else "",
                "max_fit_error_mm": round(float(np.max(finite_errors)), 3)
                if finite_errors.size
                else "",
            }
        )

        for label in segment.labels:
            data, ok, src = seg_values[label], seg_valid[label], seg_source[label]
            observed = int((src == "observed").sum())
            rigid = int((src == "rigid_fill").sum())
            data, ok, src, short = interpolate_short_gaps(
                data, ok, src, args.short_gap_frames
            )
            final_values[label] = data
            final_valid[label] = ok
            marker_rows.append(
                {
                    "marker": label,
                    "segment": segment.name,
                    "observed_frames": observed,
                    "rigid_filled_frames": rigid,
                    "short_interp_frames": short,
                    "remaining_missing_frames": int((~ok).sum()),
                    "final_visible_frames": int(ok.sum()),
                }
            )

    write_csv(args.out_dir / "segment_fit_report.csv", segment_rows)
    write_csv(args.out_dir / "marker_fill_report.csv", marker_rows)

    summary = {
        "input_c3d": str(args.c3d),
        "output_c3d": "",
        "point_frames": acq.GetPointFrameNumber(),
        "model_marker_count": len(marker_names),
        "rigid_fit_limit_mm": args.rigid_fit_limit_mm,
        "short_gap_frames": args.short_gap_frames,
        "remaining_missing_total": int(sum((~final_valid[label]).sum() for label in marker_names)),
        "fully_visible_markers": [
            label for label in marker_names if int((~final_valid[label]).sum()) == 0
        ],
    }

    if args.write_c3d:
        output_c3d = args.out_dir / f"{args.c3d.stem}_model_filled.c3d"
        for label in marker_names:
            data = np.nan_to_num(final_values[label])
            residual = np.where(final_valid[label], 0.0, -1.0).reshape((-1, 1))
            btkTools.smartAppendPoint(acq, label, data, residuals=residual)
        btkTools.smartWriter(acq, str(output_c3d))
        summary["output_c3d"] = str(output_c3d)

    (args.out_dir / "fill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
