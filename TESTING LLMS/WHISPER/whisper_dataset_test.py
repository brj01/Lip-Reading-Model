#!/usr/bin/env python3
"""
Quick test harness for transcribing audio samples listed in SCRAPE/main.json
with the OpenAI Whisper (speech-to-text) API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

from openai import OpenAI


DEFAULT_OUTPUT_PATH = Path(__file__).parent / "TESTWHISPER.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files referenced in a JSON manifest using OpenAI Whisper."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("SCRAPE/main.json"),
        help="Path to the JSON manifest containing objects with an `audio_url` field.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Optional base directory to resolve `audio_url` values. Defaults to the manifest's parent.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-transcribe",
        help="Whisper-compatible model to use (e.g. gpt-4o-mini-transcribe, whisper-1).",
    )
    parser.add_argument(
        "--language",
        default="ar",
        help="ISO 639-1 language code to steer transcription output (default: ar for Arabic).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If provided, only transcribe the first N entries from the manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to a JSON file that will store transcripts (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip persisting transcripts to disk.",
    )
    return parser.parse_args()


def load_manifest(json_path: Path) -> list[dict]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Failed to parse manifest {json_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise SystemExit(f"Manifest {json_path} must contain a JSON array.")
    return payload


def limited(entries: Iterable[dict], limit: Optional[int]) -> Iterator[dict]:
    if limit is None:
        yield from entries
        return
    for idx, entry in enumerate(entries):
        if idx >= limit:
            break
        yield entry


def resolve_audio_path(raw_url: str, base_dir: Path) -> Path:
    raw_path = Path(raw_url)
    candidates = [
        base_dir / raw_path,
        base_dir / Path(str(raw_path).replace("audios", "audio", 1)),
        base_dir / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate audio file for '{raw_url}' under {base_dir}")


def transcribe_file(client: OpenAI, audio_path: Path, model: str, language: str) -> str:
    # Writing to text keeps memory light and avoids JSON formatting noise.
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
            language=language,
        )
    if isinstance(response, str):
        return response
    # Newer SDKs return a pydantic object; `text` holds the transcript.
    return response.text  # type: ignore[attr-defined]


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] Failed to parse existing output {path}: {exc}", file=sys.stderr)
        return []
    if isinstance(data, list):
        return data
    print(f"[WARN] Existing output {path} is not a JSON array; starting fresh.", file=sys.stderr)
    return []


def upsert_record(records: list[dict], new_record: dict, key: str) -> None:
    for idx, existing in enumerate(records):
        if existing.get(key) == new_record.get(key):
            records[idx] = new_record
            return
    records.append(new_record)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.json)
    audio_base = args.audio_root or args.json.parent
    client = OpenAI()

    output_path = None if args.no_save else args.output
    output_records: list[dict] = []
    if output_path:
        output_records = load_existing_records(output_path)

    try:
        for entry in limited(manifest, args.limit):
            title = entry.get("title", "<untitled>")
            audio_url = entry.get("audio_url")
            if not audio_url:
                print(f"Skipping entry without audio_url: {title}", file=sys.stderr)
                continue

            try:
                audio_path = resolve_audio_path(audio_url, audio_base)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
                continue

            print(f"Transcribing: {title} ({audio_path})")
            try:
                transcript = transcribe_file(client, audio_path, args.model, args.language)
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[ERROR] Failed to transcribe {audio_path}: {exc}", file=sys.stderr)
                continue

            print(f"Transcript (first 120 chars): {transcript[:120]!r}")

            if output_path:
                json_record = {
                    "title": title,
                    "date": entry.get("date"),
                    "audio_url": audio_url,
                    "episode_url": entry.get("episode_url"),
                    "transcript_ar": transcript,
                }
                upsert_record(output_records, json_record, key="audio_url")
    finally:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(output_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {len(output_records)} transcript entries to {output_path}")


if __name__ == "__main__":
    main()
