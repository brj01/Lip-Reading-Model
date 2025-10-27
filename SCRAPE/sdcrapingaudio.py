import json
import os
import requests

# Paths
input_json_file = "languagewave_podcasts_with_audio.json"              # original JSON
output_json_file = "episodes_with_audio.json"  # new JSON
audio_folder = "audio"
os.makedirs(audio_folder, exist_ok=True)       # ensure folder exists

# Headers for requests (some servers block non-browser requests)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Load JSON
with open(input_json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for episode in data:
    audio_url = episode.get("audio_url")
    if not audio_url:
        continue  # skip if no audio URL

    # Determine local file path
    file_name = os.path.basename(audio_url)
    local_path = os.path.join(audio_folder, file_name)

    # Download if not already exists
    if not os.path.exists(local_path):
        try:
            with requests.get(audio_url, headers=headers, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            print(f"Downloaded: {local_path}")
        except Exception as e:
            print(f"Failed to download {audio_url}: {e}")
            continue
    else:
        print(f"Already exists: {local_path}")

    # Update JSON object
    episode["local_audio_path"] = local_path

# Save updated JSON to a new file
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"All episodes updated and saved to {output_json_file}")
