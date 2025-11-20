#!/usr/bin/env python3
"""
Utility script to transcribe a single audio file with OpenAI's
`whisper-1` API model and produce alignment-style JSON
containing sentence segments and per-word timestamps.
"""

from __future__ import annotations


import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
import os

OPENAI_CLIENT = OpenAI(api_key="cc")
def parse_payload(payload: Dict[str, Any], fallback_language: str) -> Tuple[str, List[dict], List[dict], str]:
    segments: List[dict] = []
    words_flat: List[dict] = []

    for segment in payload.get("segments") or []:
        seg_obj = {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": (segment.get("text") or "").strip(),
            "words": [],
        }
        for word in segment.get("words") or []:
            token = (word.get("word") or word.get("text") or "").strip()
            if not token:
                continue
            word_item = {
                "word": token,
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
            }
            seg_obj["words"].append(word_item)
            words_flat.append(word_item)
        segments.append(seg_obj)

    transcript = payload.get("text") or payload.get("transcript") or ""
    language = payload.get("language", fallback_language)
    return language, segments, words_flat, transcript


def transcribe(audio_path: Path, language: str) -> dict:
    with audio_path.open("rb") as audio_file:
        response = OPENAI_CLIENT.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

    payload = json.loads(response.model_dump_json())
    language, segments, words_flat, transcript = parse_payload(payload, language)
    return {
        "model": "whisper-1",
        "language": language,
        "audio_file": str(audio_path),
        "transcript_ar": transcript,
        "segments": segments,
        "words": words_flat,
        "raw": payload,
    }
def main():
    ROOT = Path(__file__).resolve().parent.parent  # repo root
    main_json_path = ROOT / "SCRAPE" / "main2.json"

    if not main_json_path.exists():
        raise SystemExit(f"ERROR: SCRAPE/main.json not found at {main_json_path}")

    data = json.loads(main_json_path.read_text(encoding="utf-8"))

    # Output directory
    out_dir = ROOT / "SCRAPE" / "ALIGN"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []  # collect episodes as we go
    final_path = out_dir / "all_episodes_whisper2.json"

    # Load existing results if file exists
    if final_path.exists():
        all_results = json.loads(final_path.read_text(encoding="utf-8"))

    for episode in data:
        # Check if this episode is already transcribed
        episode_title = episode.get("title")
        if any(r.get("title") == episode_title for r in all_results):
            print(f"SKIP: Already transcribed → {episode_title}")
            continue

        # Get audio path
        audio_rel_path = episode.get("audio_url")
        if not audio_rel_path:
            print(f"SKIP: episode missing audio_url → {episode_title}")
            continue

        audio_rel_path = Path(audio_rel_path.replace("\\", "/"))
        if not audio_rel_path.is_absolute():
            audio_path = ROOT / "SCRAPE" / audio_rel_path
        else:
            audio_path = audio_rel_path

        if not audio_path.exists():
            print(f"SKIP: Audio not found → {audio_path}")
            continue

        print(f"Transcribing: {audio_path}")
        result = transcribe(audio_path, language="ar")

        # Add episode metadata
        result["title"] = episode_title
        result["date"] = episode.get("date")
        result["original_audio_url"] = episode.get("audio_url")
        result["episode_url"] = episode.get("episode_url")

        all_results.append(result)

        # Save immediately after this episode
        final_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON after episode → {episode_title}")

    print(f"All done! Final JSON → {final_path}")


if __name__ == "__main__":
    main()
