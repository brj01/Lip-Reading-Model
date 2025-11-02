#!/usr/bin/env python3
"""
Parallel Google Cloud Speech-to-Text using GCS URIs.
Uploads audio to GCS, transcribes in Arabic (ar-LB), and saves clean results.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from google.cloud import speech, storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==== CONFIG ====
BUCKET_NAME = "example_transcript_chu_zhang"  # <-- change this to your GCS bucket
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_LANGUAGE_CODE = "ar-LB"
OUTPUT_PATH = Path("TESTING_LLMS/google_speech_to_text/test_google.json")
MAX_WORKERS = 6
# =================

lock = threading.Lock()  # for thread-safe console output


def log(msg: str):
    """Thread-safe print."""
    with lock:
        print(msg, flush=True)


def convert_to_wav(source: Path, sample_rate=DEFAULT_SAMPLE_RATE) -> Path:
    """Convert any audio file to 16kHz mono WAV."""
    if source.suffix.lower() == ".wav":
        return source

    tmp = Path(tempfile.mkstemp(prefix="google_", suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(tmp)
    ]

    # ffmpeg must be installed and available in PATH
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp


def upload_to_gcs(local_path: Path, bucket_name: str, destination_blob_name: str) -> str:
    """Upload to GCS and return the gs:// URI."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(str(local_path))
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    log(f"   [UPLOAD] {local_path.name} → {gcs_uri}")
    return gcs_uri


def transcribe_gcs(gcs_uri: str, language=DEFAULT_LANGUAGE_CODE) -> str:
    """Transcribe directly from a GCS URI."""
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=DEFAULT_SAMPLE_RATE,
        language_code=language,  # Arabic only
        enable_automatic_punctuation=True,
    )

    log(f"   [Google API] Transcribing {gcs_uri}...")
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=3600)

    # Combine all recognized text segments
    text = " ".join(r.alternatives[0].transcript.strip() for r in response.results if r.alternatives)
    return text.strip()


def process_entry(idx: int, entry: dict):
    """Convert, upload, and transcribe one audio file."""
    title = entry.get("title", f"Entry {idx}")
    audio_url = entry.get("audio_url")

    if not audio_url:
        log(f"[WARN] No audio_url for {title}")
        return None

    # Normalize Windows and Unix paths
    audio_path = Path("SCRAPE") / Path(audio_url.replace("\\", "/"))

    if not audio_path.exists():
        log(f"[WARN] Missing audio file: {audio_path}")
        return None

    log(f"[{idx}] Processing {title} ({audio_path})")

    try:
        wav_path = convert_to_wav(audio_path)
        gcs_uri = upload_to_gcs(wav_path, BUCKET_NAME, f"audio/{wav_path.name}")
        transcript = transcribe_gcs(gcs_uri, DEFAULT_LANGUAGE_CODE)
        log(f"✅ Finished {title}")
        return {
            "title": title,
            "audio_url": audio_url,
            "google_transcribed_ar": transcript
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

    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, i + 1, e): e for i, e in enumerate(manifest)}

        for future in as_completed(futures):
            result = future.result()
            if result:
                output_records.append(result)
                # Write partial results incrementally
                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                OUTPUT_PATH.write_text(
                    json.dumps(output_records, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

    log(f"\n✅ Wrote {len(output_records)} Arabic transcripts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
