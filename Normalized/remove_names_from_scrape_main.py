#!/usr/bin/env python3
"""Remove leading speaker labels from Arabic transcript fields in SCRAPE/main.json

Reads:  SCRAPE/main.json
Writes: Normalized/main.json

This script targets likely transcript keys (contains 'transcript' or 'transcrib' and 'ar')
and removes lines that start with a short speaker label followed by a colon, e.g. "رنا :",
handling optional whitespace and non-breaking spaces. It is conservative by default.
"""
from pathlib import Path
import re
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'SCRAPE' / 'main.json'
OUTPUT = ROOT / 'Normalized' / 'main.json'

# Matches a leading speaker label at the start of any line: optional whitespace,
# up to ~60 label chars (Arabic/Latin/digits/punct/space), optional spaces, colon, optional spaces.
LEADING_LABEL_RE = re.compile(
    r"(?m)^[\t\u00A0 \f\v]*[A-Za-z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9][A-Za-z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9\.\-_'\s]{0,60}?\s*:\s*",
    flags=re.UNICODE,
)


changed_count = 0
examples = []


def clean_string(s: str) -> str:
    global changed_count
    new = LEADING_LABEL_RE.sub('', s)
    if new != s:
        changed_count += 1
        if len(examples) < 12:
            examples.append({'before': s[:400].replace('\n', '\\n'), 'after': new[:400].replace('\n', '\\n')})
    return new


def should_clean_key(k: str) -> bool:
    if not k:
        return False
    k = k.lower()
    return (('transcript' in k or 'transcrib' in k) and 'ar' in k)


def process(obj):
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str):
                if should_clean_key(k):
                    obj[k] = clean_string(v)
            else:
                process(v)
    elif isinstance(obj, list):
        for item in obj:
            process(item)


def main():
    if not INPUT.exists():
        print(f"Input not found: {INPUT}")
        sys.exit(1)

    print(f"Reading: {INPUT}")
    data = json.loads(INPUT.read_text(encoding='utf-8'))

    process(data)

    out_dir = OUTPUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(OUTPUT)

    print(f"Wrote: {OUTPUT}")
    print(f"Strings changed: {changed_count}")
    if examples:
        print("Examples (before -> after, newline shown as literal '\\n'):")
        for ex in examples:
            print('- BEFORE:', ex['before'])
            print('  AFTER :', ex['after'])


if __name__ == '__main__':
    main()
