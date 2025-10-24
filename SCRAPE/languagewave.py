import os
import re
import json
import html as _html
import requests
from typing import Optional, List, Union
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup

# ====== CONFIG ======
# EITHER set EPISODE_URL to scrape a single page,
# OR set EPISODE_JSON to a JSON file containing many links (see sample JSON in the chat).
EPISODE_URL: Optional[str] = None  # e.g., "https://languagewave.com/2020/11/12/episode-51-sketch-first-date/"
EPISODE_JSON: Optional[str] = None  # e.g., "episodes.json"

OUT_BASE = "./languagewave_out"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LanguageWaveScraper/1.8; +https://example.com)"
}

# ====== CONSTANTS ======
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
MP3_RE = re.compile(r"https?://[^\s\"\']+?\.mp3(?:\?[^\s\"\']*)?", re.IGNORECASE)
_ILLEGAL_FS_CHARS = r'<>:"/\\|?*\x00-\x1F'  # for folder names

# ====== UTILS ======
_session: Optional[requests.Session] = None

def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session


def fetch(url: str) -> str:
    r = get_session().get(url, timeout=30)
    r.raise_for_status()
    return r.text


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_for_fs(name: str, max_len: int = 120) -> str:
    """
    Keep folder names human-readable; remove illegal filesystem chars across OSes.
    """
    name = name.strip()
    name = re.sub(f"[{_ILLEGAL_FS_CHARS}]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\.+$", "", name)  # no trailing dots
    return (name or "Episode")[:max_len]


def safe_mp3_filename(mp3_url: str) -> str:
    from os.path import basename, splitext
    p = urlparse(mp3_url)
    base = basename(p.path) or "audio.mp3"
    name, ext = splitext(base)
    if ext.lower() != ".mp3":
        ext = ".mp3"
    blob = parse_qs(p.query).get("blob_id", [None])[0]
    return f"{name}_{blob}{ext}" if blob else f"{name}{ext}"


def download_file(url: str, dest_path: str):
    with get_session().get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

# ====== EXTRACTION ======

def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Episode"


def content_nodes(soup: BeautifulSoup):
    article = soup.select_one("article .entry-content") or soup.select_one("article")
    return article.select("p, li") if article else soup.find_all(["p", "li"])


def extract_transcripts(soup: BeautifulSoup):
    """
    Split into Arabic vs. non-Arabic (English-ish) lines, skipping UI noise.
    """
    ar_lines, en_lines = [], []
    seen = set()
    for node in content_nodes(soup):
        text = node.get_text(" ", strip=True)
        if not text or text in seen:
            continue
        seen.add(text)
        lo = text.lower()
        if any(k in lo for k in [
            "audio player", "play", "pause", "download", "share:", "facebook", "twitter",
        ]):
            continue
        if ARABIC_RE.search(text):
            ar_lines.append(text)
        else:
            en_lines.append(text)
    return "\n".join(ar_lines).strip(), "\n".join(en_lines).strip()


def extract_first_date_mp3(html: str, soup: BeautifulSoup, base: str) -> Optional[str]:
    """
    Prefer the Buzzsprout embed: <div id="js-podcast-player" data-src="...mp3?...">
    - Unescapes &amp; → &
    - Falls back to regex/attribute search if needed
    """
    # Preferred: exact audio div
    box = soup.select_one("#js-podcast-player")
    if box:
        val = box.get("data-src")
        if val and ".mp3" in val.lower():
            return urljoin(base, _html.unescape(val))

    # Fallback: any *.mp3 in raw HTML
    m = MP3_RE.search(html)
    if m:
        return _html.unescape(m.group(0))

    # Fallback: scan attributes that may contain .mp3
    for tag in soup.find_all(["audio", "source", "a", "div", "span"]):
        for attr in ["src", "href", "data-src", "data-audio", "data-url"]:
            v = tag.get(attr)
            if v and ".mp3" in v.lower():
                return urljoin(base, _html.unescape(v))
    return None

# ====== CORE ======

def process_episode(url: str) -> None:
    print(f"==> Scraping: {url}")
    html = fetch(url)
    soup = BeautifulSoup(html, "html5lib")

    title = extract_title(soup)
    folder = ensure_dir(os.path.join(OUT_BASE, sanitize_for_fs(title)))

    # transcripts
    ar_txt, en_txt = extract_transcripts(soup)
    if ar_txt:
        with open(os.path.join(folder, "transcript_ar.txt"), "w", encoding="utf-8") as f:
            f.write(ar_txt)
        print(f"  ✓ Arabic transcript saved ({len(ar_txt)} chars)")
    else:
        print("  ! No Arabic transcript detected")

    if en_txt:
        with open(os.path.join(folder, "transcript_en.txt"), "w", encoding="utf-8") as f:
            f.write(en_txt)
        print(f"  ✓ English transcript saved ({len(en_txt)} chars)")
    else:
        print("  ! No English transcript detected")

    # MP3
    mp3 = extract_first_date_mp3(html, soup, url)
    meta = {
        "title": title,
        "url": url,
        "mp3_url": mp3,
        "arabic_chars": len(ar_txt) if ar_txt else 0,
        "english_chars": len(en_txt) if en_txt else 0,
    }

    if mp3:
        name = safe_mp3_filename(mp3)
        dest = os.path.join(folder, name)
        try:
            download_file(mp3, dest)
            print(f"  ✓ MP3 saved as {name}")
            meta["mp3_file"] = name
        except Exception as e:
            print(f"  ✗ MP3 download failed: {e}")
            meta["mp3_error"] = str(e)
    else:
        print("  ✗ No MP3 URL found on the page")

    # Save metadata for quick inspection
    with open(os.path.join(folder, "episode_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Done. Files in: {os.path.abspath(folder)}\n")


def load_episode_urls(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data: Union[list, dict] = json.load(f)
    # Accept either a plain list of URLs or an object with {"episodes": [{"url": ...}, ...]}
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "episodes" in data and isinstance(data["episodes"], list):
        urls = []
        for item in data["episodes"]:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and "url" in item:
                urls.append(str(item["url"]))
        return urls
    raise ValueError("Unsupported JSON structure for episode links")


# ====== MAIN ======

def main():
    # Allow quick CLI overrides without editing the file
    import argparse
    parser = argparse.ArgumentParser(description="LanguageWave scraper (single or batch via JSON)")
    parser.add_argument("--url", dest="url", help="Single episode URL", default=None)
    parser.add_argument("--json", dest="json_path", help="Path to JSON file of episode links", default=None)
    parser.add_argument("--out", dest="out_base", help="Output base directory", default=None)
    args = parser.parse_args()

    out_base = args.out_base or OUT_BASE
    ensure_dir(out_base)

    # override globals if provided via CLI
    url = args.url or EPISODE_URL
    json_path = args.json_path or EPISODE_JSON

    if json_path:
        urls = load_episode_urls(json_path)
        print(f"Batch mode: {len(urls)} URLs loaded from {json_path}\n")
        for u in urls:
            try:
                process_episode(u)
            except Exception as e:
                print(f"!! Error processing {u}: {e}\n")
        print("All done.")
        return

    if url:
        process_episode(url)
        return

    raise SystemExit("Provide --url or --json (path to links JSON)")


if __name__ == "__main__":
    main()
