#!/usr/bin/env python3
import json, os, re, sys, subprocess, shutil
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.resolve()
JSON_PATH = SCRIPT_DIR / "youtubevideos.json"
VIDEOS_ROOT = SCRIPT_DIR / "videos"

# Try to find ffmpeg: PATH or local ffmpeg.exe next to this script
FFMPEG = shutil.which("ffmpeg") or str((SCRIPT_DIR / "ffmpeg.exe"))

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"[\s_-]+", "_", s)
    return s or "video"

def ytdlp():
    return [sys.executable, "-m", "yt_dlp"]

def load_items():
    with open(JSON_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    items = []
    for obj in data:
        if not isinstance(obj, dict): continue
        link = obj.get("video_link") or obj.get("link") or obj.get("url")
        if not link: continue
        items.append({"name": obj.get("name") or "Untitled", "video_link": link})
    return items

def extract_mp3_from_mp4(mp4_path: Path):
    mp3_path = mp4_path.with_suffix(".mp3")
    # -vn = drop video; libmp3lame; q=2 ~ high quality
    cmd = [FFMPEG, "-y", "-i", str(mp4_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(mp3_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp3_path.exists()

def clean_srt_lines(text: str):
    out = []
    for line in text.splitlines():
        t = line.strip()
        if not t or t.isdigit() or "-->" in t:
            continue
        out.append(t)
    return "\n".join(out)

def make_transcription(out_dir: Path, base: str):
    """
    Prefer English *.srt if present; else first *.srt/.vtt.
    Write transcription.txt, then delete all *.srt/*.vtt files.
    Return True if transcription written.
    """
    srts = list(out_dir.glob("*.srt"))
    vtts = list(out_dir.glob("*.vtt"))
    if not srts and not vtts:
        return False

    def score(p: Path):
        n = p.name.lower()
        if ".en." in n or n.endswith(".en.srt") or n.endswith(".en.vtt") or n.startswith(f"{base}.en."):
            return 0
        if "en-" in n or "en_us" in n or "en-us" in n:
            return 1
        return 2

    candidates = sorted(srts + vtts, key=score)
    chosen = candidates[0]

    if chosen.suffix.lower() == ".srt":
        txt = clean_srt_lines(chosen.read_text(encoding="utf-8", errors="ignore"))
    else:
        # quick VTT clean
        lines = []
        for line in chosen.read_text(encoding="utf-8", errors="ignore").splitlines():
            t = line.strip()
            if not t or t.startswith("WEBVTT") or "-->" in t or t.startswith("NOTE") or t.startswith("STYLE"):
                continue
            lines.append(t)
        txt = "\n".join(lines)

    (out_dir / "transcription.txt").write_text(txt.strip() + ("\n" if txt else ""), encoding="utf-8")

    # remove all subtitle files so only transcription.txt remains
    for p in candidates:
        try: p.unlink()
        except Exception: pass
    return True

def delete_extras(out_dir: Path, keep: set):
    for p in out_dir.iterdir():
        if p.name not in keep:
            try: p.unlink()
            except Exception: pass

def main():
    VIDEOS_ROOT.mkdir(exist_ok=True)
    subprocess.run(ytdlp() + ["--rm-cache-dir"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    items = load_items()
    if not items:
        print("No valid items in JSON.")
        return

    for i, it in enumerate(items, 1):
        name, link = it["name"], it["video_link"]
        base = slugify(name)
        out_dir = VIDEOS_ROOT / base
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}] {name}")
        print(f"    → {link}")
        print(f"    → {out_dir}")

        # Force MP4 video and M4A audio to avoid .webm outputs
        # Limit subs to English auto only to avoid rate limiting
        cmd = ytdlp() + [
            link,
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--remux-video", "mp4",
            "--no-write-description",
            "--no-write-thumbnail",
            "--no-part",
            "--no-keep-fragments",
            "--continue",
            "-o", f"{base}.%(ext)s",
            "--socket-timeout", "5",
            "-4",

            # subtitles (optional English auto; comment these 3 lines if you don't want any)
            "--write-auto-sub",
            "--sub-langs", "en.*",
            "--convert-subs", "srt",

            # robustness
            "--retries", "10",
            "--fragment-retries", "10",
            "-N", "4",
        ]

        # If some videos 403, uncomment one line below to use your browser cookies:
        # cmd += ["--cookies-from-browser", "chrome"]   # or "edge" / "firefox"

        rc = subprocess.call(cmd, cwd=out_dir)
        if rc != 0:
            print(f"    ✖ yt-dlp failed with code {rc}")
            continue

        mp4 = out_dir / f"{base}.mp4"
        if not mp4.exists():
            print("    ⚠ MP4 not found (ffmpeg missing or blocked format).")
            continue
        print(f"    ✔ Video → {mp4}")

        # Extract MP3
        if not FFMPEG or not Path(FFMPEG).exists():
            print("    ⚠ ffmpeg not found. Place ffmpeg.exe next to this script or install ffmpeg.")
        else:
            ok = extract_mp3_from_mp4(mp4)
            if ok:
                print(f"    ✔ Audio → {mp4.with_suffix('.mp3')}")
            else:
                print("    ⚠ Failed to extract MP3.")

        # Make transcription.txt if any subs
        got_tx = make_transcription(out_dir, base)
        if got_tx:
            print(f"    ✔ Transcription → {out_dir / 'transcription.txt'}")
        else:
            print("    ℹ No subtitles available (no transcription.txt).")

        # Keep only the 3 files (or 2 if no subs)
        keep = {f"{base}.mp4", f"{base}.mp3"}
        if got_tx:
            keep.add("transcription.txt")
        delete_extras(out_dir, keep)
        print(f"    ✔ Cleaned — kept: {', '.join(sorted(keep))}")
        # --- add this just before the final print in your main() (outside the for-loop) ---
        all_mp4s = [str(p.resolve()) for p in Path(VIDEOS_ROOT).rglob("*.mp4")]
        (Path(SCRIPT_DIR) / "video_list.json").write_text(
            json.dumps(all_mp4s, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n🗂  Wrote {len(all_mp4s)} paths to video_list.json")

    # --- add this just before the final print in your main():
     
        


    print("\n✅ Done. Each folder now contains ONLY: video.mp4, audio.mp3, and transcription.txt (if available).")

if __name__ == "__main__":
    main()
