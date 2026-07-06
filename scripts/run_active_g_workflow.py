from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


SIDECAR_EXTENSIONS = [
    ".digitaldevices.xml",
    ".history",
    ".system",
    ".Trial.enf",
    ".x1d",
    ".x2d",
    ".xcp",
]

MODEL_EXTENSIONS = [".mkr", ".mp", ".vsk", ".vst"]


def run_command(command: list[str]):
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def active_c3d_path(input_c3d: Path, requested_active: Path | None) -> Path:
    if requested_active is not None:
        return requested_active
    if input_c3d.stem.lower().endswith("g"):
        return input_c3d
    return input_c3d.with_name(f"{input_c3d.stem}g.c3d")


def copy_if_possible(src: Path, dst: Path, copied: list[str], skipped: list[dict]):
    if not src.exists():
        return
    try:
        if src.resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    except OSError as exc:
        skipped.append({"source": str(src), "destination": str(dst), "reason": str(exc)})


def copy_trial_files(src_c3d: Path, dst_c3d: Path) -> dict:
    copied: list[str] = []
    skipped: list[dict] = []
    src_dir = src_c3d.parent
    dst_dir = dst_c3d.parent
    src_stem = src_c3d.stem
    dst_stem = dst_c3d.stem

    for ext in SIDECAR_EXTENSIONS:
        copy_if_possible(src_dir / f"{src_stem}{ext}", dst_dir / f"{dst_stem}{ext}", copied, skipped)

    for ext in MODEL_EXTENSIONS:
        for src in src_dir.glob(f"*{ext}"):
            copy_if_possible(src, dst_dir / src.name, copied, skipped)

    return {"copied": copied, "skipped": skipped}


def backup_file(path: Path, report_root: Path, label: str) -> Path | None:
    if not path.exists():
        return None
    report_root.mkdir(parents=True, exist_ok=True)
    backup = report_root / f"{path.stem}_{label}{path.suffix}"
    shutil.copy2(path, backup)
    return backup


def run_forward(scripts_dir: Path, input_c3d: Path, output_c3d: Path, report_dir: Path, args):
    run_command(
        [
            sys.executable,
            str(scripts_dir / "forward_connect_and_short_bridge.py"),
            "--c3d",
            str(input_c3d),
            "--model",
            str(args.model),
            "--output",
            str(output_c3d),
            "--report-dir",
            str(report_dir),
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--radius",
            str(args.radius),
            "--head-radius",
            str(args.head_radius),
            "--max-search",
            str(args.max_search),
            "--max-gap",
            "0",
            "--forward-max-mean-error",
            str(args.max_mean_error),
            "--forward-min-margin",
            str(args.min_margin),
            "--lbhd-c7-radius",
            str(args.lbhd_c7_radius),
            "--lbhd-c7-max-mean-error",
            str(args.lbhd_c7_max_mean_error),
        ]
    )


def run_reverse(scripts_dir: Path, input_c3d: Path, output_c3d: Path, report_dir: Path, args):
    run_command(
        [
            sys.executable,
            str(scripts_dir / "reverse_grey_only_headtight.py"),
            "--c3d",
            str(input_c3d),
            "--model",
            str(args.model),
            "--output",
            str(output_c3d),
            "--report-dir",
            str(report_dir),
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--radius",
            str(args.radius),
            "--head-radius",
            str(args.head_radius),
            "--max-search",
            str(args.max_search),
            "--max-mean-error",
            str(args.max_mean_error),
            "--min-margin",
            str(args.min_margin),
            "--lbhd-c7-radius",
            str(args.lbhd_c7_radius),
            "--lbhd-c7-max-mean-error",
            str(args.lbhd_c7_max_mean_error),
        ]
    )


def verify_interval(scripts_dir: Path, original_c3d: Path, final_c3d: Path, report_dir: Path, args):
    run_command(
        [
            sys.executable,
            str(scripts_dir / "verify_grey_only_result.py"),
            "--original-c3d",
            str(original_c3d),
            "--final-c3d",
            str(final_c3d),
            "--model",
            str(args.model),
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--report-dir",
            str(report_dir),
        ]
    )


def verify_full(scripts_dir: Path, original_c3d: Path, final_c3d: Path, report_dir: Path, args):
    run_command(
        [
            sys.executable,
            str(scripts_dir / "verify_full_grey_only_result.py"),
            "--original-c3d",
            str(original_c3d),
            "--final-c3d",
            str(final_c3d),
            "--model",
            str(args.model),
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--report-dir",
            str(report_dir),
        ]
    )


def suggest_manual_frames(scripts_dir: Path, c3d: Path, report_dir: Path, args):
    run_command(
        [
            sys.executable,
            str(scripts_dir / "suggest_manual_anchor_frames.py"),
            "--c3d",
            str(c3d),
            "--model",
            str(args.model),
            "--report-dir",
            str(report_dir),
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--top-n",
            str(args.manual_top_n),
            "--window",
            str(args.manual_window),
            "--min-frame-gap",
            str(args.manual_min_frame_gap),
        ]
    )


def write_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_suggest_mode(scripts_dir: Path, active_c3d: Path, report_root: Path, args):
    stage_dir = report_root / "active_stage01_suggest"
    forward_c3d = stage_dir / f"{active_c3d.stem}_forward.c3d"
    reverse_c3d = stage_dir / f"{active_c3d.stem}_suggest_ready.c3d"

    existing_backup = backup_file(active_c3d, report_root, "before_suggest_overwrite")
    run_forward(scripts_dir, args.input_c3d, forward_c3d, stage_dir / "report_forward", args)
    run_reverse(scripts_dir, forward_c3d, reverse_c3d, stage_dir / "report_reverse", args)
    verify_interval(scripts_dir, args.input_c3d, reverse_c3d, stage_dir / "report_verify", args)
    suggest_manual_frames(scripts_dir, reverse_c3d, stage_dir / "report_suggest_manual_anchor_frames", args)

    active_c3d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reverse_c3d, active_c3d)
    sidecars = copy_trial_files(args.input_c3d, active_c3d)

    summary = {
        "mode": "suggest",
        "input_c3d": str(args.input_c3d),
        "active_c3d": str(active_c3d),
        "existing_active_backup": str(existing_backup) if existing_backup else None,
        "suggested_frames_csv": str(stage_dir / "report_suggest_manual_anchor_frames" / "suggested_manual_anchor_frames.csv"),
        "sidecars": sidecars,
    }
    write_summary(stage_dir / "active_workflow_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_final_mode(scripts_dir: Path, active_c3d: Path, report_root: Path, args):
    if not active_c3d.exists():
        raise FileNotFoundError(f"Active g C3D does not exist: {active_c3d}")

    stage_dir = report_root / "active_stage02_final"
    baseline = backup_file(active_c3d, report_root, "manual_baseline")
    if baseline is None:
        raise FileNotFoundError(f"Could not back up active g C3D: {active_c3d}")

    iter1_forward = stage_dir / f"{active_c3d.stem}_iter1_forward.c3d"
    iter1 = stage_dir / f"{active_c3d.stem}_iter1.c3d"
    iter2_forward = stage_dir / f"{active_c3d.stem}_iter2_forward.c3d"
    iter2 = stage_dir / f"{active_c3d.stem}_iter2.c3d"
    outside = stage_dir / f"{active_c3d.stem}_final_checked.c3d"

    run_forward(scripts_dir, baseline, iter1_forward, stage_dir / "report_iter1_forward", args)
    run_reverse(scripts_dir, iter1_forward, iter1, stage_dir / "report_iter1_reverse", args)
    verify_interval(scripts_dir, baseline, iter1, stage_dir / "report_iter1_verify", args)

    run_forward(scripts_dir, iter1, iter2_forward, stage_dir / "report_iter2_forward", args)
    run_reverse(scripts_dir, iter2_forward, iter2, stage_dir / "report_iter2_reverse", args)
    verify_interval(scripts_dir, baseline, iter2, stage_dir / "report_iter2_verify", args)

    run_command(
        [
            sys.executable,
            str(scripts_dir / "connect_outside_from_anchor_frames.py"),
            "--c3d",
            str(iter2),
            "--model",
            str(args.model),
            "--output",
            str(outside),
            "--report-dir",
            str(stage_dir / "report_outside"),
            "--start-anchor-frame",
            str(args.start_frame),
            "--end-anchor-frame",
            str(args.end_frame),
            "--radius",
            str(args.radius),
            "--head-radius",
            str(args.head_radius),
            "--max-search",
            str(args.max_search),
            "--max-mean-error",
            str(args.max_mean_error),
            "--min-margin",
            str(args.min_margin),
            "--lbhd-c7-radius",
            str(args.lbhd_c7_radius),
            "--lbhd-c7-max-mean-error",
            str(args.lbhd_c7_max_mean_error),
        ]
    )
    verify_full(scripts_dir, baseline, outside, stage_dir / "report_verify_full", args)

    shutil.copy2(outside, active_c3d)
    summary = {
        "mode": "final",
        "baseline_backup": str(baseline),
        "active_c3d": str(active_c3d),
        "final_checked_c3d": str(outside),
        "full_verify_summary": str(stage_dir / "report_verify_full" / "verify_full_summary.json"),
    }
    write_summary(stage_dir / "active_workflow_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--mode", choices=["suggest", "final"], required=True)
    parser.add_argument("--active-c3d", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--radius", type=float, default=60.0)
    parser.add_argument("--head-radius", type=float, default=25.0)
    parser.add_argument("--lbhd-c7-radius", type=float, default=45.0)
    parser.add_argument("--lbhd-c7-max-mean-error", type=float, default=15.0)
    parser.add_argument("--max-search", type=int, default=5)
    parser.add_argument("--min-margin", type=float, default=30.0)
    parser.add_argument("--max-mean-error", type=float, default=25.0)
    parser.add_argument("--manual-top-n", type=int, default=8)
    parser.add_argument("--manual-window", type=int, default=5)
    parser.add_argument("--manual-min-frame-gap", type=int, default=20)
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    active_c3d = active_c3d_path(args.input_c3d, args.active_c3d)
    report_root = args.report_root or (active_c3d.parent / "_processing_reports")

    if args.mode == "suggest":
        run_suggest_mode(scripts_dir, active_c3d, report_root, args)
    else:
        run_final_mode(scripts_dir, active_c3d, report_root, args)


if __name__ == "__main__":
    main()
