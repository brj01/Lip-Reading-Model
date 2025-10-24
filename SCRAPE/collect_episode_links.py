import argparse, json, re, time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EpisodeLinkCollector/1.0)"}

EP_RE = re.compile(r"/20\d{2}/\d{2}/\d{2}/episode-", re.IGNORECASE)  # typical WP post url w/ 'episode-'
EP_PATH_RE = re.compile(r"/episode-\d+", re.IGNORECASE)              # looser: any '/episode-xx'

def fetch(url, timeout=30):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def is_episode_url(u: str) -> bool:
    p = urlparse(u)
    path = p.path or ""
    return bool(EP_RE.search(path) or EP_PATH_RE.search(path))

def extract_links_from_tag(tag_url: str, all_pages: bool, delay: float):
    """Walk a WordPress tag archive, follow 'Older posts' pagination, collect episode permalinks."""
    urls, seen_pages = [], set()
    page = tag_url

    def parse_page(html, base):
        soup = BeautifulSoup(html, "html.parser")
        found = set()

        # common WP structures: entry-title links + any headings with anchors
        for a in soup.select(".entry-title a, h1 a, h2 a, h3 a"):
            if a.has_attr("href"):
                found.add(urljoin(base, a["href"]))

        # fallback: any 'Continue reading' / 'Read more' links
        for a in soup.find_all("a", href=True):
            text = (a.get_text(strip=True) or "").lower()
            if "continue reading" in text or "read more" in text:
                found.add(urljoin(base, a["href"]))

        # keep only episode-looking links
        return [u for u in found if is_episode_url(u)]

    def find_next(html, base):
        soup = BeautifulSoup(html, "html.parser")
        # typical 'Older posts' / rel=next / .next.page-numbers
        for a in soup.find_all("a", href=True):
            t = (a.get_text(strip=True) or "").lower()
            if t in ("older posts", "older", "next", "next »", "older entries"):
                return urljoin(base, a["href"])
        # specialized class hooks
        a = soup.select_one("a.next, a.next.page-numbers, a[rel='next'], a.older-posts")
        if a and a.has_attr("href"):
            return urljoin(base, a["href"])
        return None

    while page and page not in seen_pages:
        seen_pages.add(page)
        html = fetch(page)
        eps = parse_page(html, page)
        for u in eps:
            if u not in urls:
                urls.append(u)
        if not all_pages:
            break
        nxt = find_next(html, page)
        if not nxt:
            break
        time.sleep(delay)
        page = nxt

    return urls

def extract_links_from_sitemap(sitemap_url: str):
    """Read sitemap.xml or sitemap index and return episode-looking URLs."""
    urls = []
    try:
        xml = fetch(sitemap_url)
    except Exception:
        return urls

    soup = BeautifulSoup(xml, "xml")

    # If it's a sitemap index, follow each <sitemap><loc>
    sitemaps = [loc.get_text(strip=True) for loc in soup.select("sitemap > loc")]
    if sitemaps:
        for sm in sitemaps:
            try:
                part_xml = fetch(sm)
                part_soup = BeautifulSoup(part_xml, "xml")
                for loc in part_soup.select("url > loc"):
                    u = loc.get_text(strip=True)
                    if is_episode_url(u):
                        urls.append(u)
            except Exception:
                continue
        return sorted(set(urls))

    # Otherwise a regular urlset
    for loc in soup.select("url > loc"):
        u = loc.get_text(strip=True)
        if is_episode_url(u):
            urls.append(u)
    return sorted(set(urls))

def main():
    ap = argparse.ArgumentParser(description="Collect LanguageWave episode URLs into a JSON file.")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g., episodes.json)")
    ap.add_argument("--tag", help="Tag/archive URL to crawl (e.g., https://languagewave.com)")
    ap.add_argument("--all-pages", action="store_true", help="Follow 'Older posts' pagination on the tag page.")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between page requests (seconds).")
    ap.add_argument("--sitemap", help="Sitemap URL (e.g., https://languagewave.com/sitemap.xml)")
    args = ap.parse_args()

    collected = []

    if args.tag:
        print(f"[tag] crawling: {args.tag}")
        collected.extend(extract_links_from_tag(args.tag, all_pages=args.all_pages, delay=args.delay))

    if args.sitemap:
        print(f"[sitemap] reading: {args.sitemap}")
        collected.extend(extract_links_from_sitemap(args.sitemap))

    # If neither provided, try a sensible default (free-sample-episodes tag)
    if not args.tag and not args.sitemap:
        default_tag = "https://languagewave.com"
        print(f"[default] crawling: {default_tag}")
        collected.extend(extract_links_from_tag(default_tag, all_pages=True, delay=args.delay))

    # de-dup while preserving order
    seen = set()
    unique_urls = []
    for u in collected:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)

    print(f"Collected {len(unique_urls)} episode URL(s).")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(unique_urls, f, ensure_ascii=False, indent=2)
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
