from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location('normalizeforWER', NORMALIZER_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load normalizeforWER module from {NORMALIZER_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

CANDIDATE_TEXT_KEYS = [
    "gpt4o_transcribed_text",
    "model_transcript",
    "transcript",
    "transcript_ar",
]

SYMBOL_MAP = {"ok": "✓", "sub": "✗", "del": "−", "ins": "+"}


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def best_text(record: Dict) -> str:
    for key in CANDIDATE_TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def build_index(records: Iterable[Dict]) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for item in records:
        audio_key = (item.get("audio_url") or "").replace("/", "\\")
        title = item.get("title") or ""
        key = audio_key or title
        if key:
            index[key] = item
    return index


def ops_to_symbols(ops: List[Tuple[str, str, str]]) -> str:
    return " | ".join(SYMBOL_MAP.get(op, "?") for op, *_ in ops)


def align_words_strict(ref_words: List[str], hyp_words: List[str]) -> List[Tuple[str, str, str]]:
    """Standard Levenshtein alignment without special variant handling."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "ins")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = ref_words[i - 1] == hyp_words[j - 1]
            substitution_cost = 0 if match else 1
            best_cost = dp[i - 1][j - 1] + substitution_cost
            best_prev = (i - 1, j - 1, "ok" if match else "sub")

            delete_cost = dp[i - 1][j] + 1
            if delete_cost < best_cost:
                best_cost = delete_cost
                best_prev = (i - 1, j, "del")

            insert_cost = dp[i][j - 1] + 1
            if insert_cost < best_cost:
                best_cost = insert_cost
                best_prev = (i, j - 1, "ins")

            dp[i][j] = best_cost
            back[i][j] = best_prev

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        assert back[i][j] is not None
        prev_i, prev_j, op = back[i][j]
        ref_word = ref_words[i - 1] if i > 0 else ""
        hyp_word = hyp_words[j - 1] if j > 0 else ""
        if op == "del":
            hyp_word = ""
        elif op == "ins":
            ref_word = ""
        ops.append((op, ref_word, hyp_word))
        i, j = prev_i, prev_j

    return list(reversed(ops))


def wer_standard(ref_text: str, hyp_text: str) -> Tuple[float, List[Tuple[str, str, str]]]:
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    ops = align_words_strict(ref_words, hyp_words)
    substitutions = sum(op == "sub" for op, *_ in ops)
    deletions = sum(op == "del" for op, *_ in ops)
    insertions = sum(op == "ins" for op, *_ in ops)
    denominator = len(ref_words)
    wer = (substitutions + deletions + insertions) / float(denominator) if denominator else float("nan")
    return wer, ops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WER between reference and model transcripts and export an alignment table."
    )
    parser.add_argument("--reference", required=True, help="Path to reference JSON (contains ground-truth transcripts).")
    parser.add_argument("--hypothesis", required=True, help="Path to model JSON (same structure, with model transcripts).")
    parser.add_argument(
        "--output",
        default="wer_alignment.csv",
        help="Output CSV path (openable in Excel). Default: wer_alignment.csv",
    )

    args = parser.parse_args()
    ref_path = Path(args.reference)
    hyp_path = Path(args.hypothesis)
    out_path = Path(args.output)

    references = load_json(ref_path)
    hypotheses = load_json(hyp_path)

    ref_index = build_index(references)

    rows: List[List[str]] = []
    missing = 0

    for hyp_entry in hypotheses:
        key = (hyp_entry.get("audio_url") or "").replace("/", "\\")
        title = hyp_entry.get("title") or key or "Untitled"
        ref_entry = ref_index.get(key) or ref_index.get(title)
        if not ref_entry:
            missing += 1
            print(f"⚠️  No reference found for '{title}' (key: '{key}')")
            continue

        ref_text = best_text(ref_entry)
        hyp_text = best_text(hyp_entry)
        wer_value, ops = wer_standard(ref_text, hyp_text)

        rows.append(
            [
                title,
                key,
                f"{wer_value:.4f}" if wer_value == wer_value else "nan",
                str(len(ref_text.split())),
                str(len(hyp_text.split())),
                ops_to_symbols(ops),
                ref_text,
                hyp_text,
            ]
        )

    header = [
        "Episode Title",
        "Audio Key",
        "WER",
        "Ref Word Count",
        "Hyp Word Count",
        "OP Sequence",
        "Reference Transcript",
        "Hypothesis Transcript",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✅ Wrote alignment summary to {out_path}")
    if missing:
        print(f"⚠️  {missing} hypothesis entries had no matching reference.")


if __name__ == "__main__":
    main()
