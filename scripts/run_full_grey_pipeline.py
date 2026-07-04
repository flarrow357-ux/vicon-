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
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)

    forward_dir = args.output_root / "stage01_forward_grey"
    final_dir = args.output_root / "stage02_reverse_once_grey"
    verify_dir = final_dir / "report_verify"
    forward_c3d = forward_dir / f"{args.final_name}_FORWARD.c3d"
    final_c3d = final_dir / f"{args.final_name}.c3d"

    copy_trial_sidecars(args.input_c3d, forward_c3d)
    run_command(
        [
            sys.executable,
            str(scripts_dir / "forward_connect_and_short_bridge.py"),
            "--c3d",
            str(args.input_c3d),
            "--model",
            str(args.model),
            "--output",
            str(forward_c3d),
            "--report-dir",
            str(forward_dir / "report"),
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

    copy_trial_sidecars(forward_c3d, final_c3d)
    run_command(
        [
            sys.executable,
            str(scripts_dir / "reverse_grey_only_headtight.py"),
            "--c3d",
            str(forward_c3d),
            "--model",
            str(args.model),
            "--output",
            str(final_c3d),
            "--report-dir",
            str(final_dir / "report"),
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

    run_command(
        [
            sys.executable,
            str(scripts_dir / "verify_grey_only_result.py"),
            "--original-c3d",
            str(args.input_c3d),
            "--final-c3d",
            str(final_c3d),
            "--model",
            str(args.model),
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--report-dir",
            str(verify_dir),
        ]
    )

    print(f"Final C3D: {final_c3d}")
    print(f"Reports: {final_dir}")


if __name__ == "__main__":
    main()

