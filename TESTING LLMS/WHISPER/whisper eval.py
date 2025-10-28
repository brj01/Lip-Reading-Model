#!/usr/bin/env python3
"""
Evaluate Whisper transcripts against cleaned ground-truth Arabic transcripts.

By default the script treats:
  • SCRAPE/clean data/output.json  as the reference set, and
  • TESTING LLMS/WHISPER/TESTWHISPER.json as the Whisper output.

It reports aggregate WER/CER/SER and highlights the worst-scoring files so you
can inspect why accuracy drops. No input files are modified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #

DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
ALEF_RE = re.compile(r"[إأآا]")
PUNCT_RE = re.compile(r"[!?،؛:….,\"«»()\[\]{}]")


def normalize_arabic(text: str) -> str:
    """Apply light normalisation so WER/CER comparisons stay consistent."""
    text = DIACRITICS_RE.sub("", text)          # remove tashkeel / diacritics
    text = ALEF_RE.sub("ا", text)               # unify alef variants
    text = text.replace("ـ", "")               # drop tatweel
    text = PUNCT_RE.sub(" ", text)             # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


# --------------------------------------------------------------------------- #
# Edit-distance metrics
# --------------------------------------------------------------------------- #

def levenshtein(a: Sequence[str], b: Sequence[str]) -> Tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) between sequences a and b."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # deletion
                    dp[i, j - 1] + 1,      # insertion
                    dp[i - 1, j - 1] + 1,  # substitution
                )

    i, j = n, m
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
            continue

        if j > 0 and (i == 0 or dp[i, j - 1] <= dp[i - 1, j] and dp[i, j - 1] <= dp[i - 1, j - 1]):
            insertions += 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i - 1, j] <= dp[i, j - 1] and dp[i - 1, j] <= dp[i - 1, j - 1]):
            deletions += 1
            i -= 1
        else:
            substitutions += 1
            i -= 1
            j -= 1
    return substitutions, deletions, insertions


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_arabic(reference).split()
    hyp_tokens = normalize_arabic(hypothesis).split()
    s, d, i = levenshtein(ref_tokens, hyp_tokens)
    n = len(ref_tokens)
    return (s + d + i) / n if n else float("nan")


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_arabic(reference).replace(" ", ""))
    hyp_chars = list(normalize_arabic(hypothesis).replace(" ", ""))
    s, d, i = levenshtein(ref_chars, hyp_chars)
    n = len(ref_chars)
    return (s + d + i) / n if n else float("nan")


def compute_ser(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    pairs = list(zip(references, hypotheses))
    if not pairs:
        return float("nan")
    mismatches = sum(
        1
        for ref, hyp in pairs
        if normalize_arabic(ref).strip() != normalize_arabic(hyp).strip()
    )
    return mismatches / len(pairs)


# --------------------------------------------------------------------------- #
# JSON loading & evaluation
# --------------------------------------------------------------------------- #

def load_transcripts(json_path: Path, key_field: str = "audio_url") -> Dict[str, str]:
    """Return {key_field: transcript_ar} mapping from a JSON list/object."""
    payload: Any = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        records: Dict[str, str] = {}
        for item in payload:
            key = str(item.get(key_field) or "").strip()
            text = item.get("transcript_ar")
            if key and isinstance(text, str):
                records[key] = text
        return records

    if isinstance(payload, dict):
        key = str(payload.get(key_field) or "").strip()
        text = payload.get("transcript_ar")
        return {key: text} if key and isinstance(text, str) else {}

    raise ValueError(f"{json_path} must be a JSON array or object.")


def print_debug_sample(ref_text: str, hyp_text: str) -> None:
    """
    Show the first 160 characters of each normalised transcript to help explain
    high error rates.
    """
    ref_norm = normalize_arabic(ref_text)
    hyp_norm = normalize_arabic(hyp_text)
    print("    ref:", ref_norm[:160])
    print("    hyp:", hyp_norm[:160])
    if ref_norm[:160] != hyp_norm[:160]:
        # Highlight rough differences in length to hint at insert/delete issues
        print(f"    len(ref)={len(ref_norm)}, len(hyp)={len(hyp_norm)}")


PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> List[str]:
    """Return non-empty paragraphs, falling back to individual lines."""
    paragraphs = [block.strip() for block in PARA_SPLIT_RE.split(text) if block.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def print_paragraphs(ref_text: str, hyp_text: str) -> None:
    """Print aligned paragraph-level snippets for reference vs. hypothesis."""
    ref_paras = split_paragraphs(ref_text)
    hyp_paras = split_paragraphs(hyp_text)
    max_len = max(len(ref_paras), len(hyp_paras))
    for idx in range(max_len):
        ref_para = ref_paras[idx] if idx < len(ref_paras) else "<missing>"
        hyp_para = hyp_paras[idx] if idx < len(hyp_paras) else "<missing>"
        print(f"    ¶{idx+1:02d} REF: {ref_para}")
        print(f"    ¶{idx+1:02d} HYP: {hyp_para}")
        if idx < max_len - 1:
            print("    ---")


def run_eval(
    reference_path: Path,
    hypothesis_path: Path,
    top_k: int,
    show_all: bool,
    quiet: bool,
) -> None:
    ref_map = load_transcripts(reference_path)
    hyp_map = load_transcripts(hypothesis_path)

    overlap = sorted(ref_map.keys() & hyp_map.keys())
    if not overlap:
        print("No overlapping entries between reference and hypothesis.")
        return

    wers, cers = [], []
    ref_sentences: List[str] = []
    hyp_sentences: List[str] = []
    per_episode: List[Tuple[str, float, float, float, Tuple[int, int, int]]] = []

    for idx, key in enumerate(overlap, start=1):
        ref_text = ref_map[key]
        hyp_text = hyp_map[key]

        wer = compute_wer(ref_text, hyp_text)
        cer = compute_cer(ref_text, hyp_text)
        ref_norm_tokens = normalize_arabic(ref_text).split()
        hyp_norm_tokens = normalize_arabic(hyp_text).split()
        sdi = levenshtein(
            ref_norm_tokens,
            hyp_norm_tokens,
        )
        ser_flag = 0.0 if normalize_arabic(ref_text).strip() == normalize_arabic(hyp_text).strip() else 1.0

        wers.append(wer)
        cers.append(cer)
        ref_sentences.append(ref_text)
        hyp_sentences.append(hyp_text)
        per_episode.append((key, wer, cer, ser_flag, sdi))

        if not quiet or show_all:
            print(f"[{idx}/{len(overlap)}] {key}")
            print(f"    WER={wer:.4f}  CER={cer:.4f}  SER={ser_flag:.4f}  (S={sdi[0]}, D={sdi[1]}, I={sdi[2]})")
            print_debug_sample(ref_text, hyp_text)
            print_paragraphs(ref_text, hyp_text)
            print()

    ser = compute_ser(ref_sentences, hyp_sentences)

    print(f"Compared {len(overlap)} matching entries.")
    print(f"WER (mean): {np.nanmean(wers):.4f}")
    print(f"CER (mean): {np.nanmean(cers):.4f}")
    print(f"SER:        {ser:.4f}")

    if quiet and not show_all:
        print(f"\nTop {min(top_k, len(per_episode))} toughest cases by WER:")
        for key, wer, cer, ser_flag, sdi in sorted(per_episode, key=lambda row: row[1], reverse=True)[:top_k]:
            print(f"- {key} | WER={wer:.4f} CER={cer:.4f} SER={ser_flag:.4f} (S={sdi[0]}, D={sdi[1]}, I={sdi[2]})")
            print_debug_sample(ref_map[key], hyp_map[key])
            print_paragraphs(ref_map[key], hyp_map[key])
            print()


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    default_ref = Path("SCRAPE/clean data/output.json")
    default_hyp = Path("TESTING LLMS/WHISPER/TESTWHISPER.json")

    parser = argparse.ArgumentParser(
        description="Compare Whisper transcripts to ground-truth Arabic transcripts."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=default_ref,
        help=f"Reference JSON path (default: {default_ref})",
    )
    parser.add_argument(
        "--hypothesis",
        type=Path,
        default=default_hyp,
        help=f"Hypothesis JSON path (default: {default_hyp})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hardest examples to print (default: 5).",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print metrics and text snippets for every overlapping entry.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode streaming output; only show aggregate stats (useful for batch runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval(args.reference, args.hypothesis, args.top_k, args.show_all, args.quiet)


if __name__ == "__main__":
    main()
