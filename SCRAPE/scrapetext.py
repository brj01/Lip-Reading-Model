import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re

# ==== CONFIG ====
INPUT_JSON = "languagewave_podcasts_with_audio.json"                     # your existing JSON file
OUTPUT_JSON = "episodes_with_transcripts.json_perfected"   # new output file
OUT_BASE = Path("./languagewave_out")

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LanguageWaveScraper/2.0)"}


def sanitize_for_fs(name: str) -> str:
    """Make safe folder names."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name).strip()[:120] or "Episode"


def fetch_page(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def extract_transcripts(html: str):
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select("article .entry-content p, article .entry-content li")
    ar_lines, en_lines = [], []
    seen = set()

    for node in nodes:
        text = node.get_text(" ", strip=True)
        if not text or text in seen:
            continue
        seen.add(text)
        lo = text.lower()
        if any(k in lo for k in ["audio player", "play", "pause", "download", "share:", "facebook", "twitter"]):
            continue
        if ARABIC_RE.search(text):
            ar_lines.append(text)
        else:
            en_lines.append(text)
    return "\n".join(ar_lines).strip(), "\n".join(en_lines).strip()


def main():
    OUT_BASE.mkdir(exist_ok=True)
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    results = []
    print(f"Loaded {len(episodes)} episodes from {INPUT_JSON}\n")

    for ep in episodes:
        url = ep.get("episode_url")
        title = ep.get("title", "Untitled")
        print(f"Scraping: {title}")

        try:
            html = fetch_page(url)
            ar_txt, en_txt = extract_transcripts(html)

            ep["transcript_ar"] = ar_txt
            ep["transcript_en"] = en_txt

            results.append(ep)

            # Stream results — update JSON progressively
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"  ✓ Added transcripts ({len(ar_txt)} Arabic chars, {len(en_txt)} English chars)\n")

        except Exception as e:
            print(f"  ✗ Error processing {url}: {e}\n")

    print(f"✅ All done. Output saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
