#!/usr/bin/env python3
"""
Parallel OpenAI GPT-4o Speech-to-Text.
Reads local audio files, transcribes in Arabic, and saves clean results.
"""

import json
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from openai import OpenAI

# ==== CONFIG ====
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_LANGUAGE_CODE = "ar"
OUTPUT_PATH = Path("TESTING_LLMS/gpt4o_transcribe/test_gpt4o.json")
MAX_WORKERS = 4  # safe number for API calls
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

lock = threading.Lock()
client = OpenAI(api_key=OPENAI_API_KEY)


def log(msg: str):
    """Thread-safe print."""
    with lock:
        print(msg, flush=True)


def convert_to_wav(source: Path, sample_rate=DEFAULT_SAMPLE_RATE) -> Path:
    """Convert any audio/video file to 16kHz mono WAV."""
    if source.suffix.lower() == ".wav":
        return source

    tmp = Path(tempfile.mkstemp(prefix="gpt4o_", suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(tmp)
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp


def transcribe_openai(audio_path: Path, language=DEFAULT_LANGUAGE_CODE) -> str:
    """Transcribe an audio file using GPT-4o."""
    log(f"   [GPT-4o] Transcribing {audio_path.name}...")

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            language=language
        )

    text = transcript.text.strip()
    return text


def process_entry(idx: int, entry: dict):
    """Convert and transcribe one audio file."""
    title = entry.get("title", f"Entry {idx}")
    audio_url = entry.get("audio_url")

    if not audio_url:
        log(f"[WARN] No audio_url for {title}")
        return None

    audio_path = Path("SCRAPE") / Path(audio_url.replace("\\", "/"))

    if not audio_path.exists():
        log(f"[WARN] Missing audio file: {audio_path}")
        return None

    log(f"[{idx}] Processing {title} ({audio_path})")

    try:
        wav_path = convert_to_wav(audio_path)
        transcript = transcribe_openai(wav_path, DEFAULT_LANGUAGE_CODE)
        log(f"✅ Finished {title}")

        return {
            "title": title,
            "audio_url": audio_url,
            "gpt4o_transcribed_text": transcript
        }

    except Exception as e:
        log(f"❌ Error in {title}: {e}")
        return None


def main():
    manifest_path = Path("SCRAPE/main.json")
    if not manifest_path.exists():
        log("❌ SCRAPE/main.json not found")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_records = []

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, i + 1, e): e for i, e in enumerate(manifest)}

        for future in as_completed(futures):
            result = future.result()
            if result:
                output_records.append(result)
                # Write partial results incrementally
                OUTPUT_PATH.write_text(
                    json.dumps(output_records, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

    log(f"\n✅ Wrote {len(output_records)} GPT-4o transcripts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
