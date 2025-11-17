#!/usr/bin/env python3
"""
Transcribe an audio file with gpt-4o-mini-transcribe and produce
alignment-style JSON with sentence and word timestamps.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with gpt-4o-mini-transcribe.")
    parser.add_argument("audio", type=Path, help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--language",
        default="ar",
        help="ISO 639-1 language code (default: ar)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path."
    )
    return parser.parse_args()


def parse_payload(payload: Dict[str, Any], fallback_language: str) -> Tuple[str, List[dict], List[dict], str]:
    segments: List[dict] = []
    words_flat: List[dict] = []

    # Each segment = sentence or paragraph
    for segment in payload.get("segments") or []:
        seg_obj = {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": (segment.get("text") or "").strip(),
            "words": [],
        }

        # Each word in the segment
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
    client = OpenAI()
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language=language,
            response_format="verbose_json",  # <-- key
            timestamp_granularities=["segment", "word"]
        )

    # Convert response to dict
    payload = json.loads(response.model_dump_json()) if hasattr(response, "model_dump_json") else json.loads(response)
    language, segments, words_flat, transcript = parse_payload(payload, language)

    return {
        "model": "gpt-4o-mini-transcribe",
        "language": language,
        "audio_file": str(audio_path),
        "transcript": transcript,
        "segments": segments,
        "words": words_flat,
        "timestamps_per_word": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words_flat],
        "timestamps_per_sentence": [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in segments]
    }


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    result = transcribe(audio_path, args.language)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        default_dir = Path("SCRAPE/ALIGN")
        default_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"align_{audio_path.stem}_gpt4o.json"
        out_path = default_dir / out_name

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved transcription to {out_path}")


if __name__ == "__main__":
    main()
