#!/usr/bin/env python3
"""
Simple orchestration entry point for the video-cleanup stage.

The script wires the existing helpers inside DATASET/VIDEOCLEANUP into a
single command so you can kick off the scene detection pass followed by the
YOLO/SyncNet refinement without hunting for individual scripts.

Example:
    python pipeline.py                       # run both passes
    python pipeline.py --skip-scene          # only run YOLO/SyncNet
    python pipeline.py --dry-run             # show the steps without running
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PYTHON = sys.executable

VIDEO_LIST = ROOT.parent / "video_list.json"
RUN_SCENE = ROOT / "run_scene_all.py"
RUN_YOLO = ROOT / "run_yolo_all.py"


def load_video_list(path: Path) -> list[str]:
    """Read and validate the shared video_list.json."""
    if not path.exists():
        raise SystemExit(f"[pipeline] Missing {path}. Run make_video_list.py first.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[pipeline] Could not parse {path}: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"[pipeline] Expected a JSON list in {path}.")
    return data


def run_step(name: str, script: Path, dry_run: bool) -> None:
    """Execute one of the helper scripts with the repo's Python."""
    if not script.exists():
        raise SystemExit(f"[pipeline] {script} not found.")
    cmd = [PYTHON, str(script)]
    print(f"[pipeline] {name}: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline wrapper for DATASET/VIDEOCLEANUP."
    )
    parser.add_argument(
        "--skip-scene",
        action="store_true",
        help="Do not run run_scene_all.py (scene detector + diarization).",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Do not run run_yolo_all.py (YOLO/SyncNet refinement).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned steps but do not execute them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = load_video_list(VIDEO_LIST)
    print(f"[pipeline] Found {len(videos)} videos in {VIDEO_LIST}")

    if not args.skip_scene:
        run_step("Scene detection", RUN_SCENE, args.dry_run)
    else:
        print("[pipeline] Skipping scene detection.")

    if not args.skip_yolo:
        run_step("YOLO / SyncNet cleanup", RUN_YOLO, args.dry_run)
    else:
        print("[pipeline] Skipping YOLO / SyncNet cleanup.")

    print("[pipeline] Done.")


if __name__ == "__main__":
    main()
