#!/usr/bin/env python3
"""Remove keys 'edits', 'diff', and 'wer' recursively from cleaned_timestamped.json.
Writes output to cleaned_no_edits.json (keeps original file intact).
Prints a small summary after processing.
"""
from pathlib import Path
import json

SRC = Path('ALIGNMENT/cleaned_timestamped.json')
OUT = Path('ALIGNMENT/cleaned_no_edits.json')
REMOVE_KEYS = {'edits', 'diff', 'wer'}


def remove_keys(obj):
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if k in REMOVE_KEYS:
                continue
            new[k] = remove_keys(v)
        return new
    if isinstance(obj, list):
        return [remove_keys(i) for i in obj]
    return obj


def main():
    if not SRC.exists():
        print(f'ERROR: source file not found: {SRC}')
        return
    data = json.loads(SRC.read_text(encoding='utf-8'))
    cleaned = remove_keys(data)
    OUT.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding='utf-8')

    # quick stats
    results = cleaned.get('results', []) if isinstance(cleaned, dict) else []
    episodes = len(results)
    total_chunks = sum(len(ep.get('chunks', [])) for ep in results)
    total_sentences = sum(len(s) for ep in results for c in ep.get('chunks', []) for s in c.get('sentences', []))
    print(f'Wrote {OUT} — episodes={episodes}, chunks={total_chunks}, sentences={total_sentences}')


if __name__ == '__main__':
    main()
