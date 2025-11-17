from __future__ import annotations

import csv
import json
from pathlib import Path

from normalizeforWER import align_words, fair_wer


ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "SCRAPE" / "main.json"
HYP_PATH = ROOT / "TESTING_LLMS" / "gpt4o_transcribe" / "test_gpt4o.json"
OUT_DIR = ROOT / "TESTING_LLMS" / "gpt4o_transcribe" / "alignment_tables"


def load_reference() -> dict[str, dict]:
    with REF_PATH.open("r", encoding="utf-8") as fh:
        episodes = json.load(fh)
    # Prefer audio_url for matching as it is stable across datasets.
    ref_by_audio = {}
    for episode in episodes:
        audio_key = (episode.get("audio_url") or "").replace("/", "\\")
        if audio_key:
            ref_by_audio[audio_key] = episode
    return ref_by_audio


def load_hypothesis() -> list[dict]:
    with HYP_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_alignment_table(title: str, ref_text: str, hyp_text: str, audio_key: str) -> None:
    ops = align_words(ref_text.split(), hyp_text.split())
    wer_value = fair_wer(ref_text, hyp_text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(audio_key).stem or title.replace(" ", "_")
    out_path = OUT_DIR / f"{safe_name}.csv"

    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Episode Title", title])
        writer.writerow(["Audio Key", audio_key])
        writer.writerow([])
        writer.writerow(["Reference Transcript"])
        writer.writerow([ref_text])
        writer.writerow([])
        writer.writerow(["Hypothesis Transcript"])
        writer.writerow([hyp_text])
        writer.writerow([])
        writer.writerow(["WER", f"{wer_value:.4f}" if wer_value == wer_value else "nan"])
        writer.writerow([])
        writer.writerow(["Operation", "Reference Word", "Hypothesis Word"])
        for op, ref_word, hyp_word in ops:
            writer.writerow([op, ref_word, hyp_word])

    print(f"Wrote alignment table → {out_path}")


def main() -> None:
    ref_by_audio = load_reference()
    hypotheses = load_hypothesis()

    missing = 0
    for record in hypotheses:
        audio_key = (record.get("audio_url") or "").replace("/", "\\")
        ref_episode = ref_by_audio.get(audio_key)
        if not ref_episode:
            missing += 1
            print(f"⚠️  No reference match for audio '{audio_key}'")
            continue

        title = record.get("title") or ref_episode.get("title") or "Untitled episode"
        ref_text = ref_episode.get("transcript_ar", "")
        hyp_text = record.get("gpt4o_transcribed_text", "")
        write_alignment_table(title, ref_text, hyp_text, audio_key)

    if missing:
        print(f"\nCompleted with {missing} unmatched hypothesis entries.")
    else:
        print("\nCompleted alignment tables for all matched episodes.")


if __name__ == "__main__":
    main()
