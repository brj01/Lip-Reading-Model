import json
import re

# Load your main JSON file
with open("main.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Regex pattern explanation:
#   (?<=\n)     → only match if there's a newline before
#   [^:\n]+     → match one or more characters that are not ':' or newline (the name)
#   \s*:\s*     → then colon (and optional spaces)
pattern = re.compile(r'(?<=\n)[^:\n]+\s*:\s*')

def clean_text(text):
    # Remove "Name :" only when preceded by newline
    return re.sub(pattern, '', text)

# Go over all items and clean "transcript_ar" if present
for item in data:
    if "transcript_ar" in item and isinstance(item["transcript_ar"], str):
        item["transcript_ar"] = clean_text(item["transcript_ar"])

# Save cleaned version
with open("data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Cleaning complete. Saved to data_cleaned.json")
