# TESTING LLMS/normalize_main.py
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from camel_tools.utils.dediac import dediac_ar

from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
    normalize_unicode
)


from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "SCRAPE" / "ALIGN" / "all_episodes_whisper1.json"
OUTPUT_PATH = ROOT / "SCRAPE" / "whisper1_normalized.json"

DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
DIGITS_RE = re.compile(r"\d+")

WEIRD_SPACES = {
    "\u00a0",  # NBSP
    "\u200f",  # RTL mark
    "\u200e",  # LTR mark
}

# ------------------------------
# NEW: Correct Arabic ligature normalizer
# ------------------------------
def normalize_ligatures(text: str) -> str:
    return (
        text.replace("ﻻ", "لا")
            .replace("ﻷ", "لأ")
            .replace("ﻹ", "لإ")
            .replace("ﻵ", "لآ")
    )


def digits_to_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        number = int(match.group(0))
        return num2words(number, lang="ar")
    return DIGITS_RE.sub(repl, text)


def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""

    # Normalize odd whitespace and canonical form
    for bad in WEIRD_SPACES:
        text = text.replace(bad, " ")
    text = unicodedata.normalize("NFC", text).strip()

    # Remove diacritics/tatweel
    text = DIACRITICS_RE.sub("", text)

    # Apply custom ligature fix BEFORE camel tools
    text = normalize_ligatures(text)

    # CAMeL normalizers
    text = normalize_unicode(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)

    # Simplify hamza variants
    text = text.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")

    # Deduplicate whitespace
    text = " ".join(text.split())

    # Remove residual diacritics (safety)
    text = dediac_ar(text)

    # Convert Western digits to Arabic words
    text = digits_to_words(text)

    return text


def normalize_entry(entry: dict) -> dict:
    normalized = dict(entry)
    for field in ("title", "transcript_ar", "description"):
        raw = entry.get(field, "")
        normalized[f"{field}_normalized"] = normalize_arabic_text(raw)
    return normalized


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as fh:
        episodes = json.load(fh)

    normalized = [normalize_entry(ep) for ep in episodes]

    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(normalized, fh, ensure_ascii=False, indent=2)

    print(f"Normalized {len(normalized)} episodes → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()