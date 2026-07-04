from __future__ import annotations

import argparse
import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from viconnexusapi import ViconNexus


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
    order = []
    values = {}
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
    segments = []
    for segment in order:
        coords = []
        for index in sorted(values[segment]):
            axes = values[segment][index]
            coords.append([axes["x"], axes["y"], axes["z"]])
        labels = marker_names[cursor : cursor + len(coords)]
        if len(labels) != len(coords):
            break
        cursor += len(coords)
        segments.append(SegmentModel(segment, labels, np.asarray(coords)))
    return segments


def rigid_transform(model_points, observed_points):
    model_centroid = model_points.mean(axis=0)
    observed_centroid = observed_points.mean(axis=0)
    h = (model_points - model_centroid).T @ (observed_points - observed_centroid)
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = observed_centroid - model_centroid @ rotation.T
    return rotation, translation


def write_pipeline(path: Path, frame: int):
    xml = f"""<?xml version="1.1" encoding="UTF-8" standalone="no" ?>
<Pipeline>
  <Entry DisplayName="Autolabel Static Frame" Enabled="1" OperationId="25" OperationName="TPoseLabel">
    <ParamList name="" version="1">
      <Param name="FRAME" value="{frame}"/>
      <Param name="Tolerance" value="0.025"/>
      <Param name="SeparationDistance" value="1"/>
      <Param name="ClearLabels" value="1"/>
      <Param macro="ACTIVE_SUBJECTS" name="SUBJECTS"/>
    </ParamList>
  </Entry>
</Pipeline>
"""
    path.write_text(xml, encoding="utf-8")


def get_marker_data(nexus, subject, marker_names):
    values = {}
    valid = {}
    for marker in marker_names:
        try:
            x, y, z, exists = nexus.GetTrajectory(subject, marker)
        except Exception:
            x, y, z, exists = [], [], [], []
        if not exists:
            frame_count = nexus.GetFrameCount()
            values[marker] = np.full((frame_count, 3), np.nan)
            valid[marker] = np.zeros(frame_count, dtype=bool)
            continue
        ok = np.asarray(exists, dtype=bool)
        data = np.column_stack([x, y, z]).astype(float)
        data[~ok] = np.nan
        values[marker] = data
        valid[marker] = ok
    return values, valid


def evaluate_segments(segments, values, valid):
    rows = []
    reliable = 0
    weighted_error = 0.0
    weighted_frames = 0
    for segment in segments:
        errors = []
        usable = 0
        frame_count = next(iter(values.values())).shape[0]
        for frame in range(frame_count):
            present = [i for i, label in enumerate(segment.labels) if valid[label][frame]]
            if len(present) < 3:
                continue
            model_obs = segment.coords[present]
            data_obs = np.asarray([values[segment.labels[i]][frame] for i in present])
            rotation, translation = rigid_transform(model_obs, data_obs)
            reconstructed = model_obs @ rotation.T + translation
            error = float(np.sqrt(np.mean(np.sum((reconstructed - data_obs) ** 2, axis=1))))
            errors.append(error)
            if error <= 35.0:
                usable += 1
        errors_np = np.asarray(errors)
        median_error = float(np.median(errors_np)) if errors_np.size else float("nan")
        p95_error = float(np.percentile(errors_np, 95)) if errors_np.size else float("nan")
        if usable >= frame_count * 0.4 and np.isfinite(median_error) and median_error <= 35.0:
            reliable += 1
        if np.isfinite(median_error):
            weighted_error += median_error * max(1, usable)
            weighted_frames += max(1, usable)
        rows.append(
            {
                "segment": segment.name,
                "usable_frames": usable,
                "median_error_mm": round(median_error, 3) if np.isfinite(median_error) else "",
                "p95_error_mm": round(p95_error, 3) if np.isfinite(p95_error) else "",
            }
        )
    score = reliable * 100000 + weighted_frames - weighted_error
    return rows, reliable, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vst", required=True, type=Path)
    parser.add_argument("--mkr", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shared-pipeline", required=True, type=Path)
    parser.add_argument("--pipeline-name", default="temp_AutoInitializeProbe")
    parser.add_argument("--frames", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    marker_names = parse_marker_names(args.mkr)
    segments = parse_vst_segments(args.vst, marker_names)
    nexus = ViconNexus.ViconNexus()
    subject = nexus.GetSubjectNames()[0]

    summary_rows = []
    detail_rows = []
    for frame in [int(item) for item in args.frames.split(",") if item.strip()]:
        write_pipeline(args.shared_pipeline, frame)
        time.sleep(0.1)
        nexus.RunPipeline(args.pipeline_name, "Shared", 120)
        values, valid = get_marker_data(nexus, subject, marker_names)
        segment_rows, reliable_count, score = evaluate_segments(segments, values, valid)
        visible_total = int(sum(valid[label].sum() for label in marker_names))
        summary_rows.append(
            {
                "frame": frame,
                "reliable_segments": reliable_count,
                "visible_total": visible_total,
                "score": round(score, 3),
                "unlabeled_count": nexus.GetUnlabeledCount(),
            }
        )
        for row in segment_rows:
            row = dict(row)
            row["frame"] = frame
            detail_rows.append(row)
        print(json.dumps(summary_rows[-1], ensure_ascii=False))

    with (args.out_dir / "auto_init_frame_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.out_dir / "auto_init_frame_segments.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)


if __name__ == "__main__":
    main()
