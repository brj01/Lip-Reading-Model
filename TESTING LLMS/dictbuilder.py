#!/usr/bin/env python3
"""
Generate alternative spellings for Arabic words in paragraph chunks
(using ChatGPT API) and stream results into JSON as they are generated.
"""

import json
from pathlib import Path
import openai
import time
import re

# ==== CONFIG ====

MAIN_JSON = Path("SCRAPE/main.json")
OUTPUT_JSON = Path("SCRAPE/paragraph_word_alternatives_chunked.json")
MODEL_NAME = "gpt-4o-mini"  # or "gpt-4o" for higher accuracy
MAX_WORDS_PER_CHUNK = 100
RATE_LIMIT_DELAY = 1  # seconds between requests
# =================

openai.api_key = OPENAI_API_KEY


def parse_json_array_from_text(text: str):
    """Extract JSON array from GPT response text."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def chunk_paragraph(paragraph: str, max_words=MAX_WORDS_PER_CHUNK):
    """
    Split a long paragraph into smaller chunks of at most `max_words` words.
    Returns a list of paragraph chunks.
    """
    words = paragraph.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def get_paragraph_chunks_from_manifest(manifest_path: Path):
    """Extract all transcript chunks (≤100 words) from the manifest JSON."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = []
    for entry in manifest:
        transcript = entry.get("transcript_ar", "").strip()
        if not transcript:
            continue
        for chunk in chunk_paragraph(transcript):
            chunks.append(chunk)
    return chunks


def get_alternatives_for_paragraph_chunk(paragraph_chunk: str):
    """
    Given a paragraph chunk, return a JSON array of dictionaries.
    Each dictionary maps a word to its valid alternative spellings.
    """

    prompt = f"""
أنت خبير في اللغة العربية وعلم اللهجات.
سأعطيك فقرة (أو جزء فقرة) باللغة العربية تحتوي على كلمات فصحى ولهجية وأحيانًا كلمات منقولة من لغات أجنبية.
مهمتك هي أن تُعيد قائمة JSON من القواميس، كل قاموس يحتوي الكلمة كمفتاح ومجموعة بدائلها كقيمة.

مثال للإخراج المطلوب:
[
  {{"كلمة": ["كلمة"]}},
  {{"شوكولا": ["شوكولا", "شوكولاته", "شوكولاه"]}},
  {{"تويتر": ["تويتر", "توتر"]}}
]

التعليمات الدقيقة:
1. إذا كانت الكلمة عربية فصحى أو لها تهجئة قياسية واحدة → أعدها كما هي فقط.
2. إذا كانت لهجية → أعد حتى 5 تهجئات دارجة مستخدمة، بشرط أن يكون المعنى والنطق نفسه.
3. إذا كانت أجنبية أو منقولة صوتيًا → أعد حتى 5 تهجئات شائعة لنفس اللفظ.
4. لا تغيّر المعنى أو تضيف لاحقات أو بادئات (مثل ال، ت، ها، نا، ـكم، ـه، ـش...).
5. تجاهل الاختلافات في:
   - التشكيل والحركات
   - كتابة الألف (ا / أ / إ / آ / ى)
   - كتابة الياء (ي / ى)
   - المسافات أو العلامات
6. لا تعيد الجملة بصيغتها، فقط أعد JSON نظيف يحتوي على الكلمات وبدائلها.
7. حافظ على ترتيب الكلمات كما ظهرت في الفقرة.

الفقرة:
\"\"\"{paragraph_chunk}\"\"\"
"""

    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()
        parsed = parse_json_array_from_text(content)
        if parsed:
            return parsed
    except Exception as e:
        print(f"⚠️ Error generating alternatives for paragraph chunk: {e}")

    # fallback: return basic mapping (no change)
    words = paragraph_chunk.split()
    return [{w: [w]} for w in words]


def main():
    chunks = get_paragraph_chunks_from_manifest(MAIN_JSON)
    print(f"Found {len(chunks)} paragraph chunks (≤{MAX_WORDS_PER_CHUNK} words each).")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    results = []
    # Resume from existing output
    if OUTPUT_JSON.exists():
        try:
            results = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            print(f"Resuming from {len(results)} chunks already processed...")
        except Exception as e:
            print(f"⚠️ Could not read existing JSON: {e}")

    start_index = len(results)

    for idx, paragraph_chunk in enumerate(chunks[start_index:], start=start_index + 1):
        print(f"[{idx}/{len(chunks)}] Processing paragraph chunk ({len(paragraph_chunk.split())} words)...")

        alts = get_alternatives_for_paragraph_chunk(paragraph_chunk)
        results.append({
            "chunk_text": paragraph_chunk,
            "word_alternatives": alts
        })

        # Stream to JSON after each chunk
        OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\n✅ Done! Results written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
