from __future__ import annotations

import argparse
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


def copy_trial_sidecars(src_c3d: Path, dst_c3d: Path):
    src_dir = src_c3d.parent
    dst_dir = dst_c3d.parent
    src_base = src_c3d.stem
    dst_base = dst_c3d.stem
    dst_dir.mkdir(parents=True, exist_ok=True)

    for ext in SIDECAR_EXTENSIONS:
        src = src_dir / f"{src_base}{ext}"
        if src.exists():
            shutil.copy2(src, dst_dir / f"{dst_base}{ext}")

    for ext in MODEL_EXTENSIONS:
        for src in src_dir.glob(f"*{ext}"):
            shutil.copy2(src, dst_dir / src.name)


def run_command(command):
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def run_forward(scripts_dir: Path, input_c3d: Path, output_c3d: Path, report_dir: Path, args):
    copy_trial_sidecars(input_c3d, output_c3d)
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
    copy_trial_sidecars(input_c3d, output_c3d)
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


def run_interval_verify(scripts_dir: Path, original_c3d: Path, final_c3d: Path, report_dir: Path, args):
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


def run_full_verify(scripts_dir: Path, original_c3d: Path, final_c3d: Path, report_dir: Path, args):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-c3d", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--final-name", default="FINAL_GREY_ONLY")
    parser.add_argument("--radius", type=float, default=60.0)
    parser.add_argument("--head-radius", type=float, default=25.0)
    parser.add_argument("--lbhd-c7-radius", type=float, default=45.0)
    parser.add_argument("--lbhd-c7-max-mean-error", type=float, default=15.0)
    parser.add_argument("--max-search", type=int, default=5)
    parser.add_argument("--min-margin", type=float, default=30.0)
    parser.add_argument("--max-mean-error", type=float, default=25.0)
    parser.add_argument("--suggest-manual-frames", action="store_true")
    parser.add_argument("--stop-after-suggestion", action="store_true")
    parser.add_argument("--manual-top-n", type=int, default=8)
    parser.add_argument("--manual-window", type=int, default=5)
    parser.add_argument("--manual-min-frame-gap", type=int, default=20)
    parser.add_argument("--second-iteration", action="store_true")
    parser.add_argument("--connect-outside", action="store_true")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)

    iter1_forward_dir = args.output_root / "stage01_iter1_forward_grey"
    iter1_final_dir = args.output_root / "stage02_iter1_reverse_grey"
    iter1_forward_c3d = iter1_forward_dir / f"{args.final_name}_ITER1_FORWARD.c3d"
    iter1_c3d = iter1_final_dir / f"{args.final_name}_ITER1.c3d"

    run_forward(scripts_dir, args.input_c3d, iter1_forward_c3d, iter1_forward_dir / "report", args)
    run_reverse(scripts_dir, iter1_forward_c3d, iter1_c3d, iter1_final_dir / "report", args)
    run_interval_verify(scripts_dir, args.input_c3d, iter1_c3d, iter1_final_dir / "report_verify", args)

    if args.suggest_manual_frames:
        suggest_dir = iter1_final_dir / "report_suggest_manual_anchor_frames"
        run_command(
            [
                sys.executable,
                str(scripts_dir / "suggest_manual_anchor_frames.py"),
                "--c3d",
                str(iter1_c3d),
                "--model",
                str(args.model),
                "--report-dir",
                str(suggest_dir),
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
        print(f"Manual frame suggestions: {suggest_dir / 'suggested_manual_anchor_frames.csv'}")
        if args.stop_after_suggestion:
            print(f"Stopped after suggestion. Manually label the suggested frames in: {iter1_c3d}")
            return

    current_c3d = iter1_c3d
    current_dir = iter1_final_dir

    if args.second_iteration:
        iter2_dir = args.output_root / "stage03_iter2_forward_reverse_grey"
        iter2_forward_c3d = iter2_dir / f"{args.final_name}_ITER2_FORWARD.c3d"
        iter2_c3d = iter2_dir / f"{args.final_name}_ITER2.c3d"
        run_forward(scripts_dir, current_c3d, iter2_forward_c3d, iter2_dir / "report_forward", args)
        run_reverse(scripts_dir, iter2_forward_c3d, iter2_c3d, iter2_dir / "report_reverse", args)
        run_interval_verify(scripts_dir, args.input_c3d, iter2_c3d, iter2_dir / "report_verify", args)
        current_c3d = iter2_c3d
        current_dir = iter2_dir

    if args.connect_outside:
        outside_dir = args.output_root / "stage04_connect_outside"
        outside_c3d = outside_dir / f"{args.final_name}.c3d"
        copy_trial_sidecars(current_c3d, outside_c3d)
        run_command(
            [
                sys.executable,
                str(scripts_dir / "connect_outside_from_anchor_frames.py"),
                "--c3d",
                str(current_c3d),
                "--model",
                str(args.model),
                "--output",
                str(outside_c3d),
                "--report-dir",
                str(outside_dir / "report"),
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
        run_full_verify(scripts_dir, args.input_c3d, outside_c3d, outside_dir / "report_verify_full", args)
        current_c3d = outside_c3d
        current_dir = outside_dir

    print(f"Final C3D: {current_c3d}")
    print(f"Reports: {current_dir}")


if __name__ == "__main__":
    main()
