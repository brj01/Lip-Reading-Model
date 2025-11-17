import json
import os
import time
import unicodedata
from openai import OpenAI

# === CONFIG ===
INPUT_FILE = "SCRAPE/main.json"
OUTPUT_FILE = "episodes_with_transliterations.json"
RATE_LIMIT_DELAY = 2  # seconds between requests to avoid rate limits
MODEL = "gpt-4o"  # or whichever you use

raw_key = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY = raw_key.replace("\xa0", " ").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def normalize_text(text: str) -> str:
    """Normalize text and replace bad spaces."""
    if not text:
        return ""
    # Replace all weird whitespace with normal spaces
    text = text.replace("\xa0", " ").replace("\u200f", "").replace("\u200e", "")
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def generate_transliterations(word: str) -> list[str]:
    """Ask GPT for possible transliterations."""
    word = normalize_text(word)

    # Ensure fully UTF-8 safe before sending to API
    safe_word = word.encode("utf-8", errors="ignore").decode("utf-8")

    prompt = (
        f"Generate up to 5 possible transliterations for the Arabic or dialectal word '{safe_word}'. "
        "If the word is already standard Arabic, keep it as is. "
        "If it’s a foreign or English word written in Arabic, return likely Latin spellings. "
        "Return only the transliterations as a JSON array of strings."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()

        try:
            translits = json.loads(content)
            if isinstance(translits, list):
                return [normalize_text(t) for t in translits]
            else:
                return [normalize_text(content)]
        except Exception:
            return [normalize_text(content)]

    except Exception as e:
        print(f"⚠️ Error generating transliterations for '{word}': {e}")
        return []


def process_episodes(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Loaded {total} items from {input_path}")

    for i, entry in enumerate(data, start=1):
        title = entry.get("title", "")
        print(f"[{i}/{total}] Processing '{title}'...")

        transcript = normalize_text(entry.get("transcript_ar", ""))
        if not transcript:
            entry["transliterations"] = []
            continue

        words = set(transcript.split())
        translit_dict = {}

        for word in words:
            translit_dict[word] = generate_transliterations(word)
            time.sleep(RATE_LIMIT_DELAY)

        entry["transliterations"] = translit_dict

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved updated file to {output_path}")


if __name__ == "__main__":
    process_episodes(INPUT_FILE, OUTPUT_FILE)