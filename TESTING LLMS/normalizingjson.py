import json
import os
import re
import time
from num2words import num2words

# === CONFIG ===
INPUT_FILE = r"ALIGNMENT\cleaned_no_edits.json"
OUTPUT_FILE = r"ALIGNMENT\cleanednormalized.json"

# === REGEX + DIGIT NORMALIZATION ===
def normalize_arabic_regex(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # --- Arabic-specific normalization ---
    # Normalize alef variants
    text = re.sub(r"[إأآٱ]", "ا", text)

    # Normalize yaa and alef maksura
    text = re.sub(r"ى", "ي", text)

    # Normalize teh marbuta → ه
    text = re.sub(r"ة", "ه", text)

    # Remove tatweel (ـ)
    text = re.sub(r"ـ", "", text)

    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]", "", text)

    # Remove invisible / control characters
    text = text.replace("\u200f", "").replace("\u200e", "").replace("\xa0", " ")

    # Remove punctuation (Arabic + English) but keep English letters
    punctuation_pattern = r"[\"'،؛؟!?,.;:()\[\]{}<>~`@#$%^&*_+=\\|/−—–]"
    text = re.sub(punctuation_pattern, " ", text)

    # Keep Arabic, English, digits, and whitespace — remove symbols only
    text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF\s]", " ", text)

    # Convert Western digits (0-9) to Arabic-Indic digits (٠-٩)
    western_to_arabic_digits = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    text = text.translate(western_to_arabic_digits)

    # Convert numbers to Arabic words
    text = convert_numbers_to_text(text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def convert_numbers_to_text(text: str) -> str:
    """Convert any Arabic or Western digit sequence into Arabic words."""
    def replacer(match):
        num_str = match.group()
        # Convert Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to Western for num2words
        western_num = num_str.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        try:
            num = int(western_num)
            return num2words(num, lang="ar")
        except Exception:
            return num_str  # fallback if conversion fails

    return re.sub(r"[0-9٠-٩]+", replacer, text)


# === Recursive JSON normalization ===
def normalize_json(data, parent_key=None):
    if isinstance(data, dict):
        normalized = {}
        for k, v in data.items():
            # Skip normalization for title or audio keys
            if k.lower() in ["title", "audio", "audio_link","audio_url"] or "audio" in k.lower():
                normalized[k] = v
            else:
                normalized[k] = normalize_json(v, parent_key=k)
        return normalized

    elif isinstance(data, list):
        return [normalize_json(item, parent_key) for item in data]

    elif isinstance(data, str):
        # Only normalize if Arabic or digits exist in text
        if re.search(r"[\u0600-\u06FF0-9٠-٩]", data):
            return normalize_arabic_regex(data)
        return data

    else:
        return data


# === Main processing ===
def process_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If file has a top-level wrapper like { "results": [...] }, unwrap it.
    if isinstance(data, dict) and "results" in data:
        data = data["results"]

    # Ensure we have a list to iterate
    if not isinstance(data, list):
        data = [data]

    total = len(data)
    print(f"📖 Loaded {total} items from {input_path}")

    normalized_entries = []
    for i, raw_entry in enumerate(data, start=1):
        entry = raw_entry
        # If the item is a JSON string, try to parse it into an object.
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception:
                # Not a JSON object — treat as plain text under key 'text'
                entry = {"text": raw_entry}

        # Safely get a title for logging if the entry is a dict
        if isinstance(entry, dict):
            title = entry.get("title", entry.get("text", ""))
        else:
            title = ""

        print(f"[{i}/{total}] Normalizing '{title}'...")
        normalized_entry = normalize_json(entry)
        normalized_entries.append(normalized_entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_entries, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Normalized file saved to {output_path}")


if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)
