import json
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright
import time

# Paths
input_file = "languagewave_podcasts.json"
output_file = "languagewave_podcasts_with_audio.json"
download_folder = Path("audios")
download_folder.mkdir(exist_ok=True)
def wait_for_user():
    print("You have 1 minute to log in manually. Press 'q' then Enter when ready...")
    start_time = time.time()
    while True:
        if time.time() - start_time > 60:
            print("1 minute elapsed. Starting scraping...")
            break
        if input().lower() == 'q':
            print("User pressed 'q'. Starting scraping...")
            break

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    episodes = json.load(f)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    wait_for_user()
    for ep in episodes:
        url = ep["episode_url"]
        print(f"Scraping: {url}")
        try:
            page.goto(url)
            time.sleep(2)  # wait a bit for audio to load

            # Get <audio> src
            audio_url = page.eval_on_selector("audio", "el => el.src") if page.query_selector("audio") else None
            if audio_url:
                print("Audio found:", audio_url)
                filename = download_folder / audio_url.split("/")[-1].split("?")[0]

                # Download audio
                r = requests.get(audio_url, stream=True)
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                ep["audio_url"] = str(filename)
                print("Downloaded:", filename)
            else:
                print("No audio found.")
                ep["audio_url"] = None

        except Exception as e:
            print("Error:", e)
            ep["audio_url"] = None

    browser.close()

# Save new JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(episodes, f, indent=2, ensure_ascii=False)

print(f"All done! Saved updated JSON to {output_file}")
