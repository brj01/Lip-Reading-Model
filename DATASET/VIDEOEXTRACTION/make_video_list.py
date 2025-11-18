#!/usr/bin/env python3
import json, os
from pathlib import Path

VIDEOS_ROOT = Path(__file__).parent.resolve() / "videos"

def main():
    paths = [str(p.resolve()) for p in VIDEOS_ROOT.rglob("*.mp4")]
    out = Path(__file__).parent / "video_list.json"
    out.write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(paths)} entries → {out}")

if __name__ == "__main__":
    main()
