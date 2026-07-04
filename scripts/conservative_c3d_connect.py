from __future__ import annotations

import argparse
import csv
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class C3DLayout:
    labels: list[str]
    first_frame: int
    last_frame: int
    point_used: int
    data_start: int
    scale: float
    point_rate: float
    analog_used: int
    analog_rate: float
    stride: int
    base: int


def _decode_param(dtype: int, dims: list[int], raw: bytes):
    if dtype == -1:
        if len(dims) == 2:
            width, count = dims
            return [
                raw[i * width : (i + 1) * width].decode("latin1", "ignore").strip()
                for i in range(count)
            ]
        return raw.decode("latin1", "ignore")
    if dtype == 2:
        values = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
    elif dtype == 4:
        values = list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    elif dtype == 1:
        values = list(raw)
    else:
        values = list(raw)
    return values[0] if len(values) == 1 else values


def parse_params(data: bytes) -> dict[tuple[str, str], tuple[int, list[int], bytes]]:
    start = (data[0] - 1) * 512
    pos = start + 4
    group_names: dict[int, str] = {}
    params: dict[tuple[str, str], tuple[int, list[int], bytes]] = {}

    while pos < len(data):
        if data[pos] == 0:
            break
        raw_len = struct.unpack("b", data[pos : pos + 1])[0]
        group_id = struct.unpack("b", data[pos + 1 : pos + 2])[0]
        name_len = abs(raw_len)
        name = data[pos + 2 : pos + 2 + name_len].decode("latin1").strip()
        offset_pos = pos + 2 + name_len
        offset = struct.unpack("<h", data[offset_pos : offset_pos + 2])[0]
        content = offset_pos + 2
        next_pos = offset_pos + offset

        if group_id < 0:
            group_names[abs(group_id)] = name.upper()
        else:
            dtype = struct.unpack("b", data[content : content + 1])[0]
            ndims = data[content + 1]
            dims = list(data[content + 2 : content + 2 + ndims])
            item_count = 1
            for dim in dims:
                item_count *= dim
            item_size = {-1: 1, 1: 1, 2: 2, 4: 4}.get(abs(dtype), 1)
            data_start = content + 2 + ndims
            raw = data[data_start : data_start + item_count * item_size]
            params[(group_names.get(group_id, str(group_id)), name.upper())] = (
                dtype,
                dims,
                raw,
            )

        if next_pos <= pos:
            break
        pos = next_pos
    return params


def load_c3d(path: Path):
    data = bytearray(path.read_bytes())
    params = parse_params(data)
    labels = _decode_param(*params[("POINT", "LABELS")])
    point_used = int(_decode_param(*params[("POINT", "USED")]))
    first_frame = int(_decode_param(*params[("TRIAL", "ACTUAL_START_FIELD")])[0])
    last_frame = int(_decode_param(*params[("TRIAL", "ACTUAL_END_FIELD")])[0])
    data_start = int(_decode_param(*params[("POINT", "DATA_START")]))
    scale = float(_decode_param(*params[("POINT", "SCALE")]))
    point_rate = float(_decode_param(*params[("POINT", "RATE")]))
    analog_used = int(_decode_param(*params.get(("ANALOG", "USED"), (2, [1], b"\0\0"))))
    analog_rate = float(_decode_param(*params.get(("ANALOG", "RATE"), (4, [1], b"\0\0\0\0"))))

    if scale >= 0:
        raise ValueError("This conservative connector currently supports floating-point C3D point data only.")

    analog_samples = int(round(analog_rate / point_rate)) if point_rate else 0
    stride = point_used * 16 + analog_used * analog_samples * 4
    base = (data_start - 1) * 512
    expected_end = base + (last_frame - first_frame + 1) * stride
    if expected_end != len(data):
        raise ValueError(f"C3D data layout mismatch: expected {expected_end}, file has {len(data)} bytes")

    frame_count = last_frame - first_frame + 1
    xyz = np.zeros((frame_count, point_used, 3), dtype=float)
    residual = np.zeros((frame_count, point_used), dtype=float)
    for frame_i in range(frame_count):
        frame_base = base + frame_i * stride
        for point_i in range(point_used):
            off = frame_base + point_i * 16
            x, y, z, r = struct.unpack("<ffff", data[off : off + 16])
            xyz[frame_i, point_i] = [x, y, z]
            residual[frame_i, point_i] = r

    layout = C3DLayout(
        labels=labels,
        first_frame=first_frame,
        last_frame=last_frame,
        point_used=point_used,
        data_start=data_start,
        scale=scale,
        point_rate=point_rate,
        analog_used=analog_used,
        analog_rate=analog_rate,
        stride=stride,
        base=base,
    )
    return data, layout, xyz, residual


def write_points(data: bytearray, layout: C3DLayout, xyz: np.ndarray, residual: np.ndarray, output: Path):
    for frame_i in range(xyz.shape[0]):
        frame_base = layout.base + frame_i * layout.stride
        for point_i in range(layout.point_used):
            off = frame_base + point_i * 16
            data[off : off + 16] = struct.pack(
                "<ffff",
                float(xyz[frame_i, point_i, 0]),
                float(xyz[frame_i, point_i, 1]),
                float(xyz[frame_i, point_i, 2]),
                float(residual[frame_i, point_i]),
            )
    output.write_bytes(data)


def load_model(path: Path):
    names: list[str] = []
    edges: list[tuple[str, str]] = []
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("!") or text.startswith("["):
            continue
        if "," in text:
            left, right = [part.strip() for part in text.split(",", 1)]
            if left and right:
                edges.append((left, right))
        elif re.match(r"^[A-Z][A-Z0-9]*$", text):
            names.append(text)
    return names, edges


def is_valid(xyz: np.ndarray, residual: np.ndarray):
    return (residual != -1.0) & np.isfinite(xyz).all(axis=2) & ~(np.isclose(xyz, 0.0).all(axis=2))


def marker_step_limit(marker: str) -> float:
    if marker in {"RASIS", "LASIS", "RPSIS", "LPSIS", "CLAV", "C7", "STRN", "T10"}:
        return 65.0
    if marker.endswith("HD") or marker in {"LSHO", "RSHO", "LTROC", "RTROC"}:
        return 75.0
    if marker in {"LFIN", "RFIN", "LWRB", "RWRB", "LWRA", "RWRA", "LDW", "RDW"}:
        return 105.0
    if marker in {"LHM1", "LHM2", "LHM5", "RHM1", "RHM2", "RHM5", "LHEEL", "RHEEL"}:
        return 105.0
    return 90.0


def support_requirement(marker: str) -> int:
    if marker in {"LFIN", "RFIN", "LWRB", "RWRB", "LWRA", "RWRA", "LDW", "RDW"}:
        return 2
    if marker in {"RBHD", "LBHD", "RFHD", "LFHD", "CLAV", "C7", "STRN", "T10"}:
        return 2
    return 1


def build_reference_lengths(
    xyz: np.ndarray,
    valid: np.ndarray,
    label_to_i: dict[str, int],
    edges: list[tuple[str, str]],
    start_i: int,
    end_i: int,
):
    refs: dict[tuple[str, str], float] = {}
    for a, b in edges:
        if a not in label_to_i or b not in label_to_i:
            continue
        ai, bi = label_to_i[a], label_to_i[b]
        lengths = []
        for frame_i in (start_i, end_i):
            if valid[frame_i, ai] and valid[frame_i, bi]:
                lengths.append(float(np.linalg.norm(xyz[frame_i, ai] - xyz[frame_i, bi])))
        if lengths:
            refs[(a, b)] = float(np.mean(lengths))
            refs[(b, a)] = float(np.mean(lengths))
    return refs


def neighbor_map(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    neighbors: dict[str, list[str]] = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    return neighbors


def predict_position(xyz: np.ndarray, valid: np.ndarray, frame_i: int, marker_i: int, direction: int):
    prev = frame_i - direction
    prev2 = frame_i - 2 * direction
    pred = xyz[prev, marker_i].copy()
    if 0 <= prev2 < xyz.shape[0] and valid[prev2, marker_i]:
        velocity = xyz[prev, marker_i] - xyz[prev2, marker_i]
        if np.linalg.norm(velocity) <= marker_step_limit(""):
            pred = pred + velocity
    return pred


def candidate_score(
    marker: str,
    candidate_pos: np.ndarray,
    pred: np.ndarray,
    xyz_work: np.ndarray,
    valid_work: np.ndarray,
    frame_i: int,
    refs: dict[tuple[str, str], float],
    neighbors: dict[str, list[str]],
    label_to_i: dict[str, int],
):
    pred_dist = float(np.linalg.norm(candidate_pos - pred))
    limit = marker_step_limit(marker)
    if pred_dist > limit:
        return None

    support = 0
    seg_errors: list[float] = []
    for nb in neighbors.get(marker, []):
        nb_i = label_to_i.get(nb)
        if nb_i is None or not valid_work[frame_i, nb_i]:
            continue
        ref = refs.get((marker, nb))
        if ref is None:
            continue
        dist = float(np.linalg.norm(candidate_pos - xyz_work[frame_i, nb_i]))
        error = abs(dist - ref)
        tol = max(35.0, min(75.0, ref * 0.24))
        if error <= tol:
            support += 1
            seg_errors.append(error)
        else:
            seg_errors.append(error + 100.0)

    if support < support_requirement(marker):
        return None
    mean_seg_error = float(np.mean(seg_errors)) if seg_errors else 999.0
    if mean_seg_error > 45.0:
        return None

    # If the candidate is almost on top of an already-valid body marker, it is not a safe unlabeled point.
    body_indices = [point_i for label, point_i in label_to_i.items() if not label.startswith("*")]
    body_positions = xyz_work[frame_i, body_indices]
    body_valid = valid_work[frame_i, body_indices]
    if np.any(body_valid):
        distances = np.linalg.norm(body_positions[body_valid] - candidate_pos, axis=1)
        distances = distances[distances > 1.0]
        nearest_body = float(np.min(distances)) if distances.size else 999.0
        if nearest_body < 18.0:
            return None

    score = pred_dist + mean_seg_error * 1.4 - support * 4.0
    return score, pred_dist, mean_seg_error, support


def run_direction(
    direction: int,
    xyz: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
    labels: list[str],
    marker_names: list[str],
    unlabeled: list[str],
    label_to_i: dict[str, int],
    refs: dict[tuple[str, str], float],
    neighbors: dict[str, list[str]],
    start_i: int,
    end_i: int,
):
    xyz_work = xyz.copy()
    residual_work = residual.copy()
    valid_work = valid.copy()
    assignments: dict[tuple[int, str], dict] = {}

    if direction == 1:
        frames = range(start_i + 1, end_i)
    else:
        frames = range(end_i - 1, start_i, -1)

    unlabeled_i = [labels.index(label) for label in unlabeled]
    for frame_i in frames:
        prev = frame_i - direction
        missing = [
            marker
            for marker in marker_names
            if not valid_work[frame_i, label_to_i[marker]] and valid_work[prev, label_to_i[marker]]
        ]
        available = [label for label in unlabeled if valid_work[frame_i, labels.index(label)]]
        if not missing or not available:
            continue

        candidates: list[tuple[int, int, dict]] = []
        for mi, marker in enumerate(missing):
            marker_i = label_to_i[marker]
            pred = predict_position(xyz_work, valid_work, frame_i, marker_i, direction)
            scored = []
            for ui, raw_label in enumerate(available):
                raw_i = labels.index(raw_label)
                result = candidate_score(
                    marker,
                    xyz_work[frame_i, raw_i],
                    pred,
                    xyz_work,
                    valid_work,
                    frame_i,
                    refs,
                    neighbors,
                    label_to_i,
                )
                if result is None:
                    continue
                score, pred_dist, seg_error, support = result
                scored.append((score, raw_label, pred_dist, seg_error, support))
            scored.sort(key=lambda item: item[0])
            if not scored:
                continue
            best = scored[0]
            second = scored[1][0] if len(scored) > 1 else best[0] + 999.0
            if second - best[0] < 18.0:
                continue
            raw_label = best[1]
            ui = available.index(raw_label)
            candidates.append(
                (
                    mi,
                    ui,
                    {
                        "marker": marker,
                        "raw": raw_label,
                        "score": best[0],
                        "pred_dist": best[2],
                        "seg_error": best[3],
                        "support": best[4],
                        "margin": second - best[0],
                    },
                )
            )

        if not candidates:
            continue
        cost = np.full((len(missing), len(available)), 1e6, dtype=float)
        meta: dict[tuple[int, int], dict] = {}
        for mi, ui, info in candidates:
            cost[mi, ui] = info["score"]
            meta[(mi, ui)] = info
        row_ind, col_ind = linear_sum_assignment(cost)

        for mi, ui in zip(row_ind, col_ind):
            if cost[mi, ui] >= 1e6:
                continue
            info = meta[(int(mi), int(ui))]
            marker = info["marker"]
            raw_label = info["raw"]
            marker_i = label_to_i[marker]
            raw_i = labels.index(raw_label)
            xyz_work[frame_i, marker_i] = xyz_work[frame_i, raw_i]
            residual_work[frame_i, marker_i] = residual_work[frame_i, raw_i]
            valid_work[frame_i, marker_i] = True
            xyz_work[frame_i, raw_i] = 0.0
            residual_work[frame_i, raw_i] = -1.0
            valid_work[frame_i, raw_i] = False
            assignments[(frame_i, marker)] = info

    return assignments


def merge_assignments(forward: dict, backward: dict, first_frame: int):
    accepted: list[dict] = []
    review: list[dict] = []
    pending_chain: list[tuple[tuple[int, str], dict, str]] = []
    keys = sorted(set(forward) | set(backward))
    for key in keys:
        f = forward.get(key)
        b = backward.get(key)
        frame_i, marker = key
        if f and b and f["raw"] == b["raw"]:
            info = dict(f)
            info["direction"] = "both"
            accepted.append({"frame": frame_i + first_frame, **info})
            continue
        strong = f or b
        if strong and strong["pred_dist"] <= 32.0 and strong["seg_error"] <= 22.0 and strong["support"] >= 2 and strong["margin"] >= 40.0:
            info = dict(strong)
            info["direction"] = "forward_strong" if f else "backward_strong"
            accepted.append({"frame": frame_i + first_frame, **info})
        elif strong and strong["pred_dist"] <= 16.0 and strong["seg_error"] <= 34.0 and strong["support"] >= 3 and strong["margin"] >= 100.0:
            pending_chain.append((key, dict(strong), "forward_chain" if f else "backward_chain"))
        else:
            item = {"frame": frame_i + first_frame, "marker": marker}
            if f:
                item.update({f"forward_{k}": v for k, v in f.items()})
            if b:
                item.update({f"backward_{k}": v for k, v in b.items()})
            review.append(item)

    pending_chain.sort(key=lambda item: (item[0][1], item[2], item[0][0]))
    chain: list[tuple[tuple[int, str], dict, str]] = []

    def flush_chain():
        if len(chain) < 5:
            for key, info, direction in chain:
                frame_i, marker = key
                review.append(
                    {
                        "frame": frame_i + first_frame,
                        "marker": marker,
                        f"{direction}_raw": info["raw"],
                        f"{direction}_pred_dist": info["pred_dist"],
                        f"{direction}_seg_error": info["seg_error"],
                        f"{direction}_support": info["support"],
                        f"{direction}_margin": info["margin"],
                    }
                )
            return
        for key, info, direction in chain:
            frame_i, _marker = key
            item = dict(info)
            item["direction"] = direction
            accepted.append({"frame": frame_i + first_frame, **item})

    prev_key = None
    prev_direction = None
    for key, info, direction in pending_chain:
        same_chain = (
            prev_key is not None
            and key[1] == prev_key[1]
            and direction == prev_direction
            and key[0] == prev_key[0] + 1
        )
        if not same_chain and chain:
            flush_chain()
            chain = []
        chain.append((key, info, direction))
        prev_key = key
        prev_direction = direction
    if chain:
        flush_chain()
    return accepted, review


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=2127)
    parser.add_argument("--end-frame", type=int, default=3041)
    args = parser.parse_args()

    data, layout, xyz, residual = load_c3d(args.c3d)
    marker_names, edges = load_model(args.model)
    label_to_i = {label: i for i, label in enumerate(layout.labels)}
    missing = [marker for marker in marker_names if marker not in label_to_i]
    if missing:
        raise ValueError(f"Model markers missing from C3D labels: {missing}")

    valid = is_valid(xyz, residual)
    start_i = args.start_frame - layout.first_frame
    end_i = args.end_frame - layout.first_frame
    for frame_i, label in [(start_i, "start"), (end_i, "end")]:
        missing_frame = [m for m in marker_names if not valid[frame_i, label_to_i[m]]]
        if missing_frame:
            raise ValueError(f"{label} frame is not complete: {args.start_frame if label == 'start' else args.end_frame} missing {missing_frame}")

    unlabeled = [label for label in layout.labels if label.startswith("*")]
    refs = build_reference_lengths(xyz, valid, label_to_i, edges, start_i, end_i)
    neighbors = neighbor_map(edges)

    forward = run_direction(
        1, xyz, residual, valid, layout.labels, marker_names, unlabeled, label_to_i, refs, neighbors, start_i, end_i
    )
    backward = run_direction(
        -1, xyz, residual, valid, layout.labels, marker_names, unlabeled, label_to_i, refs, neighbors, start_i, end_i
    )
    accepted, review = merge_assignments(forward, backward, layout.first_frame)

    xyz_out = xyz.copy()
    residual_out = residual.copy()
    valid_out = valid.copy()
    for item in accepted:
        frame_i = int(item["frame"]) - layout.first_frame
        marker_i = label_to_i[item["marker"]]
        raw_i = label_to_i[item["raw"]]
        if valid_out[frame_i, marker_i] or not valid_out[frame_i, raw_i]:
            continue
        xyz_out[frame_i, marker_i] = xyz_out[frame_i, raw_i]
        residual_out[frame_i, marker_i] = residual_out[frame_i, raw_i]
        xyz_out[frame_i, raw_i] = 0.0
        residual_out[frame_i, raw_i] = -1.0
        valid_out[frame_i, marker_i] = True
        valid_out[frame_i, raw_i] = False

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_points(data, layout, xyz_out, residual_out, args.output)
    write_csv(args.report_dir / "accepted_connections.csv", accepted)
    write_csv(args.report_dir / "review_not_connected.csv", review)

    before_missing = {
        marker: int(np.sum(~valid[start_i : end_i + 1, label_to_i[marker]]))
        for marker in marker_names
    }
    after_valid = is_valid(xyz_out, residual_out)
    after_missing = {
        marker: int(np.sum(~after_valid[start_i : end_i + 1, label_to_i[marker]]))
        for marker in marker_names
    }
    summary = [
        {
            "marker": marker,
            "before_missing": before_missing[marker],
            "after_missing": after_missing[marker],
            "connected": before_missing[marker] - after_missing[marker],
        }
        for marker in marker_names
        if before_missing[marker] or before_missing[marker] != after_missing[marker]
    ]
    write_csv(args.report_dir / "missing_summary.csv", summary)

    print(f"accepted={len(accepted)}")
    print(f"review_not_connected={len(review)}")
    print(f"output={args.output}")
    print(f"report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
