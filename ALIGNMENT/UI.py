#!/usr/bin/env python3
"""
create_karaoke_video.py

Create an MP4 that shows word-level highlighting (karaoke) from WhisperX word-level timestamps.

Usage:
  1) (optional) Run whisperx yourself:
       whisperx path/to/audio.wav --model small --highlight_words True --align_model wav2vec2
     or let this script call whisperx (requires `whisperx` on PATH).

  2) Produce word-level JSON (whisperx produces segments with a "words" list).
     Expected JSON structure (per segment):
     [
       {
         "start": 0.0,
         "end": 2.3,
         "text": "i want the code for the video",
         "words": [
           {"word": "i", "start": 0.0, "end": 0.15},
           {"word": "want", "start": 0.15, "end": 0.55},
           ...
         ]
       },
       ...
     ]

  3) Run:
       python create_karaoke_video.py --audio path/to/audio.wav --words path/to/words.json --out sample01.mp4

Requirements:
  - ffmpeg installed and in PATH (needed to render the wave and burn ASS subtitles)
  - whisperx on PATH if using --run-whisperx
  - Python 3.7+
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from typing import List, Dict, Any


def format_ass_time(t: float) -> str:
    # ASS time format: H:MM:SS.CS  (centiseconds)
    total_seconds = int(math.floor(t))
    cs = int(round((t - math.floor(t)) * 100))
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ass(text: str) -> str:
    # Basic escaping for ASS: remove braces and newlines which break ASS markup
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").replace("\r", "")


def make_ass_from_word_segments(segments: List[Dict[str, Any]], width=1280, height=720) -> str:
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "Timer: 100.0000",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        # PrimaryColour is &HAABBGGRR: &H00FFFFFF => white opaque
        "Style: Default,Arial,44,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,50,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    events = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        words = seg.get("words", [])
        if not words:
            # fallback: use whole segment text as static line (no karaoke)
            text = escape_ass(seg.get("text", "")).strip()
            if not text:
                continue
            dialogue_text = text
        else:
            pieces = []
            # For each word, compute duration in centiseconds relative to the word's own start/end
            for w in words:
                w_word = w.get("word", "").strip()
                if w_word == "":
                    continue
                # Use word start/end if available, otherwise approximate inside the segment
                w_start = float(w.get("start", seg_start))
                w_end = float(w.get("end", min(seg_end, w_start + 0.05)))
                dur_seconds = max(0.01, w_end - w_start)
                dur_cs = int(round(dur_seconds * 100))
                # Add karaoke tag for this word. Keep a trailing space so words are separated.
                pieces.append(r"{\k%d}%s" % (dur_cs, escape_ass(w_word)))
            # Join using spaces so the ASS renderer shows word gaps
            dialogue_text = " ".join(pieces)

        start_ass = format_ass_time(seg_start)
        end_ass = format_ass_time(seg_end)
        events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{dialogue_text}")

    return "\n".join(header + events)


def load_words_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both { "segments": [...] } and list-of-segments
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected JSON structure for words/segments.")


def run_whisperx(audio_path: str, out_dir: str, model: str = "small", device: str = "cuda") -> str:
    """
    Run whisperx to produce alignment. This helper will place files in out_dir and return
    the path to the first JSON it finds.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "whisperx",
        audio_path,
        "--model",
        model,
        "--device",
        device,
        "--highlight_words",
        "True",
        "--output_dir",
        out_dir,
        "--align_model",
        "wav2vec2",
    ]
    print("Running whisperx:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)
    for fn in os.listdir(out_dir):
        if fn.endswith(".json"):
            return os.path.join(out_dir, fn)
    raise FileNotFoundError("No JSON output found in whisperx output directory.")


def build_and_render(video_out: str, audio_in: str, ass_path: str, width=1280, height=720):
    # Create ffmpeg filter_complex:
    # 1) audio -> showwaves visualisation
    # 2) burn ASS subtitles onto the video
    filter_complex = (
        f"[0:a]aformat=channel_layouts=stereo,showwaves=s={width}x{height}:mode=cline:colors=White[v];"
        f"[v]ass={shlex.quote(ass_path)}[vout]"
    )
    ff_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_in,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "0:a",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        video_out,
    ]
    print("Running ffmpeg to render video:")
    print(" ".join(shlex.quote(x) for x in ff_cmd))
    subprocess.run(ff_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Create karaoke-style video from WhisperX word timings.")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav/mp3).")
    parser.add_argument("--words", required=False, help="Path to whisperx word-level JSON (or segments JSON).")
    parser.add_argument("--out", default="sample01.mp4", help="Output MP4 file.")
    parser.add_argument("--ass", default="subtitle.ass", help="Temporary ASS subtitle path to write.")
    parser.add_argument("--run-whisperx", action="store_true", help="If set, run whisperx on the audio first (requires whisperx CLI).")
    parser.add_argument("--whisperx-outdir", default="whisperx_out", help="Output dir to store whisperx results if --run-whisperx is used.")
    parser.add_argument("--model", default="small", help="Whisper model to use if running whisperx.")
    parser.add_argument("--device", default="cuda", help="Device for whisperx (cuda/cpu).")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    args = parser.parse_args()

    if args.run_whisperx:
        if args.words:
            print("--run-whisperx set but --words also provided; using provided --words instead of running whisperx.")
        else:
            try:
                words_json = run_whisperx(args.audio, args.whisperx_outdir, model=args.model, device=args.device)
                print("WhisperX produced:", words_json)
                args.words = words_json
            except Exception as e:
                print("Failed to run whisperx:", e, file=sys.stderr)
                sys.exit(1)

    if not args.words or not os.path.exists(args.words):
        print("ERROR: --words JSON is required (output from whisperx with word timestamps).", file=sys.stderr)
        parser.print_help()
        sys.exit(2)

    try:
        segments = load_words_json(args.words)
    except Exception as e:
        print("Failed to load words JSON:", e, file=sys.stderr)
        sys.exit(1)

    ass_text = make_ass_from_word_segments(segments, width=args.width, height=args.height)
    with open(args.ass, "w", encoding="utf-8") as f:
        f.write(ass_text)
    print(f"Wrote ASS subtitles to {args.ass}")

    try:
        build_and_render(args.out, args.audio, args.ass, width=args.width, height=args.height)
        print(f"Wrote video to {args.out}")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()