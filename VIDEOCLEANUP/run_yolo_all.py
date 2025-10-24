#!/usr/bin/env python3
import json, math, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VIDEO_LIST = ROOT.parent / "video_list.json"
SCRIPT = ROOT / "auto_single_yolo.py"

BATCH_TAG = "batch1"  # appears in output folder names
TOTAL_JOBS = 1
JOB_INDEX  = 0

def main():
    videos = json.loads(VIDEO_LIST.read_text(encoding="utf-8"))
    n = len(videos)
    if n == 0:
        print("No videos found in video_list.json"); return

    per = math.ceil(n / TOTAL_JOBS)
    start = JOB_INDEX * per
    end   = min(start + per, n)
    this_chunk = videos[start:end]
    print(f"YOLO/SyncNet on {len(this_chunk)} / {n} videos (idx {start}:{end})")

    tmp_list = ROOT / f"_video_list_yolo_{JOB_INDEX}.json"
    tmp_list.write_text(json.dumps(this_chunk, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [sys.executable, str(SCRIPT), "--file_path", str(tmp_list), "--batch_num", BATCH_TAG]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
