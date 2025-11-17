from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "SCRAPE" / "main.json"
OUTPUT_PATH = ROOT / "SCRAPE" / "enriched" / "main_with_vocab.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_vocab_fallback(html: str) -> list[str]:
    """Fallback BeautifulSoup parser when Playwright lookup fails."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "lebanese arabic" not in headers:
            continue
        words: list[str] = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                word = cols[1].get_text(strip=True)
                if word:
                    words.append(word)
        return words
    return []


def enrich_episode(page, episode: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(episode)

    episode_url = episode.get("episode_url")
    if not episode_url:
        enriched["vocab"] = []
        return enriched

    for attempt in range(3):
        try:
            page.goto(episode_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_load_state("networkidle", timeout=30_000)
            break
        except PlaywrightTimeoutError as exc:
            if attempt == 2:
                print(f"   ! navigation timed out → {exc}")
                enriched["vocab"] = []
                return enriched
            print("   ! navigation retrying...")
            page.wait_for_timeout(1500)

    time.sleep(1.5)

    try:
        table_locator = page.locator("table:has(th:has-text('Lebanese Arabic'))")
        table_locator.wait_for(timeout=60_000)
        arabic_cells = table_locator.locator("tbody tr td:nth-child(2)")
        vocab = [cell.inner_text().strip() for cell in arabic_cells.all()]
    except PlaywrightTimeoutError:
        vocab = extract_vocab_fallback(page.content())

    enriched["vocab"] = vocab
    return enriched


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as fh:
        episodes = json.load(fh)

    print(f"Loaded {len(episodes)} episodes from {INPUT_PATH}")

    enriched_episodes: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()

        input("Log in if needed, then press Enter to continue...")

        for idx, episode in enumerate(episodes, start=1):
            title = episode.get("title", "Untitled")
            print(f"[{idx}/{len(episodes)}] {title}")
            try:
                enriched_episodes.append(enrich_episode(page, episode))
            except Exception as exc:  # noqa: BLE001
                print(f"   ! failed to scrape → {exc}")
                failed = dict(episode)
                failed["vocab"] = []
                enriched_episodes.append(failed)

        browser.close()

    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(enriched_episodes, fh, ensure_ascii=False, indent=2)

    print(f"\nSaved enriched data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()



