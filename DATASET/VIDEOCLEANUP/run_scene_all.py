#!/usr/bin/env python3
import json, math, subprocess, sys
from pathlib import Path

# paths
ROOT = Path(__file__).parent.resolve()
VIDEO_LIST = ROOT.parent / "video_list.json"   # adjust if needed
SCRIPT = ROOT / "auto_single_scene.py"         # your existing file

# config
TOTAL_JOBS = 1   # set >1 if you want to split work into multiple passes
JOB_INDEX  = 0   # run 0..TOTAL_JOBS-1 to process its chunk

def main():
    videos = json.loads(VIDEO_LIST.read_text(encoding="utf-8"))
    n = len(videos)
    if n == 0:
        print("No videos found in video_list.json"); return

    per = math.ceil(n / TOTAL_JOBS)
    start = JOB_INDEX * per
    end   = min(start + per, n)
    this_chunk = videos[start:end]
    print(f"Processing {len(this_chunk)} / {n} videos (idx {start}:{end})")

    # write a temp list for this chunk
    tmp_list = ROOT / f"_video_list_{JOB_INDEX}.json"
    tmp_list.write_text(json.dumps(this_chunk, ensure_ascii=False, indent=2), encoding="utf-8")

    # call your script
    cmd = [sys.executable, str(SCRIPT), "--file_path", str(tmp_list), "--idx", "0", "--num", "1"]
    # ↑ --idx/--num semantics follow your existing script:
    #    it reads the JSON list and processes num items starting from idx
    #    If your auto_single_scene.py processes the WHOLE list when given a file_path,
    #    simply remove the two flags: ["--file_path", tmp_list]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
