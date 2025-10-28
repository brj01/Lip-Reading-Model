#!/usr/bin/env python3
"""Utility for stripping speaker labels from Arabic transcripts in main.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Regex pattern to detect speaker labels like "مريم:" or "المديرة :"
# (must appear at the beginning of a line, not mid-sentence)
LABEL_PATTERN = re.compile(
    r"(?m)^\s*"  # line start + optional spaces
    r"(?:"
    r"[\u0600-\u06FFA-Za-z0-9\u0660-\u0669\u0640'._-]+"  # first token
    r"(?:\s+[\u0600-\u06FFA-Za-z0-9\u0660-\u0669\u0640'._-]+){0,3}"  # up to 3 more
    r")\s*"
    r"[:：]\s*"  # colon (Arabic/Latin/fullwidth) + optional whitespace
)

DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "main.json"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_PATH.with_name("main_clean.json")


def clean_speaker_labels(text: str) -> str:
    """Remove speaker names such as 'مريم:' or 'نعيم :' from the start of lines."""
    if not text:
        return text
    return LABEL_PATTERN.sub("", text)


def clean_json(input_path: Path, output_path: Path) -> None:
    """Read JSON, clean transcript_ar fields, and write the cleaned data."""
    with input_path.open("r", encoding="utf-8") as source_file:
        data: Any = json.load(source_file)

    # Normalize list/single-object structure
    episodes = data if isinstance(data, list) else [data]

    for episode in episodes:
        transcript = episode.get("transcript_ar")
        if isinstance(transcript, str):
            episode["transcript_ar"] = clean_speaker_labels(transcript)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target_file:
        payload = episodes if isinstance(data, list) else episodes[0]
        json.dump(payload, target_file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean transcript_ar fields by removing leading speaker labels."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the input JSON (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write the cleaned JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_json(args.input, args.output)
    print(f"✅ Cleaned transcripts saved to: {args.output}")


if __name__ == "__main__":
    main()

