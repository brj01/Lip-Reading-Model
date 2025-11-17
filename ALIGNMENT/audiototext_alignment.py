#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instant transcription of SCRAPE/main.json episodes with per-word timestamps.

For each episode:
 - Reads audio from audio_url (relative to SCRAPE/main.json)
 - Calls the Whisper API (word timestamps enabled) via OpenAI
 - Saves immediately as SCRAPE/align_<title>.json after finishing
 - Does not wait for all files to finish before saving
 - Original SCRAPE/main.json is not modified

Usage:
  OPENAI_API_KEY=... python instant_transcribe_scrape.py --json SCRAPE/main.json --lang ar --model whisper-1
"""

import os, json, argparse, re
from pathlib import Path
from datetime import datetime
from openai import OpenAI

def sanitize_filename(name: str) -> str:
    """Clean episode title for filesystem-safe filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def transcribe(audio_path: Path, client: OpenAI, language="ar", model_name="whisper-1"):
    """
    Call the Whisper API with timestamp_granularities=['word'] and shape the response.
    """
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            language=language,
            response_format="verbose_json",
            temperature=0,
            timestamp_granularities=["word"]
        )

    data = json.loads(response.model_dump_json())

    segments = []
    words_flat = []
    transcript = []

    for seg in data.get("segments", []) or []:
        seg_obj = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text", "").strip(),
            "words": []
        }
        if seg_obj["text"]:
            transcript.append(seg_obj["text"])
        for w in seg.get("words") or []:
            token = (w.get("word") or w.get("text") or "").strip()
            if token and "start" in w and "end" in w:
                word_item = {
                    "word": token,
                    "start": float(w["start"]),
                    "end": float(w["end"])
                }
                seg_obj["words"].append(word_item)
                words_flat.append(word_item)
        segments.append(seg_obj)

    return {
        "language": data.get("language", language),
        "segments": segments,
        "words": words_flat,
        "transcript_ar": "\n".join(transcript).strip()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to SCRAPE/main.json")
    parser.add_argument("--lang", default="ar", help="Language code (default: ar)")
    parser.add_argument("--model", default="whisper-1", help="Whisper API model name")
    args = parser.parse_args()

    json_path = Path(args.json)
    base_dir = json_path.parent
    out_dir = base_dir / "ALIGN"
    out_dir.mkdir(exist_ok=True)

    try:
        episodes = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    try:
        client = OpenAI()
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    for idx, ep in enumerate(episodes):
        title = ep.get("title", f"Episode_{idx+1}")
        audio_url = ep.get("audio_url")
        if not audio_url:
            print(f"[skip] {title}: no audio_url")
            continue

        audio_path = base_dir / audio_url.replace("\\", os.sep)
        if not audio_path.exists():
            print(f"[missing] {title}: {audio_path}")
            continue

        safe_title = sanitize_filename(title)
        out_file = out_dir / f"align_{safe_title}.json"
        if out_file.exists():
            print(f"[exists] {out_file} already, skipping.")
            continue

        print(f"[Transcribing] {title}...")
        try:
            result = transcribe(audio_path, client, language=args.lang, model_name=args.model)
        except Exception as e:
            print(f"[error] {title}: {e}")
            continue

        output = {
            "title": title,
            "date": ep.get("date"),
            "audio_url": audio_url,
            "episode_url": ep.get("episode_url"),
            "transcript_ar": result.get("transcript_ar", ""),
            "transcript_en": ep.get("transcript_en", ""),
            "segments": result.get("segments", []),
            "words": result.get("words", []),
            "saved_at": datetime.now().isoformat()
        }

        out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Saved {out_file}")

    print("\nAll done!")

if __name__ == "__main__":
    main()
