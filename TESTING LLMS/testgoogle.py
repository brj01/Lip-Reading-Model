#!/usr/bin/env python3
"""
Batch transcription helper for Google Cloud Speech-to-Text (Arabic focus).

This mirrors the Whisper/Munsit/Parakeet harnesses:
  * Loads a JSON manifest (default: SCRAPE/main.json) with `audio_url` records
  * Resolves local audio files
  * Sends them to Google Cloud Speech-to-Text (sync or async)
  * Persists transcripts to TESTING LLMS/testgoogle.json by default

Before running:
  * Install dependencies: `pip install google-cloud-speech google-cloud-storage` (storage only needed when using GCS)
  * Set GOOGLE_APPLICATION_CREDENTIALS to the path of your service-account JSON key, or authenticate via gcloud
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from google.cloud import speech


DEFAULT_OUTPUT_PATH = Path(__file__).with_suffix(".json")
DEFAULT_LANGUAGE = "ar"
DEFAULT_MODEL = "latest_long"
DEFAULT_ENCODING = speech.RecognitionConfig.AudioEncoding.LINEAR16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files referenced in a JSON manifest using Google Speech-to-Text."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("SCRAPE/main.json"),
        help="Path to JSON manifest containing objects with an `audio_url` field (default: SCRAPE/main.json).",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Optional base directory to resolve audio paths; defaults to the manifest's parent directory.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code for recognition (default: {DEFAULT_LANGUAGE}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Speech-to-Text model hint (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--encoding",
        default="LINEAR16",
        help="AudioEncoding value (LINEAR16, FLAC, MP3, OGG_OPUS, etc.). Defaults to LINEAR16.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz for the audio (default: 16000).",
    )
    parser.add_argument(
        "--max-alternatives",
        type=int,
        default=1,
        help="Number of alternative transcripts to request (default: 1).",
    )
    parser.add_argument(
        "--punctuation",
        action="store_true",
        help="Enable automatic punctuation.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include word-level timestamps in the saved JSON output.",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use long-running recognize (recommended for audio > 1 min).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5400,
        help="Timeout in seconds to wait for async operations (default: 5400 = 90 minutes).",
    )
    parser.add_argument(
        "--gcs-bucket",
        help="Upload audio to this GCS bucket before recognition (required for large files).",
    )
    parser.add_argument(
        "--gcs-prefix",
        default="speech-batch/",
        help="Prefix (folder) inside the GCS bucket when uploading (default: speech-batch/).",
    )
    parser.add_argument(
        "--keep-gcs",
        action="store_true",
        help="Do not delete uploaded GCS objects after transcription.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert audio to LINEAR16 16 kHz mono WAV via ffmpeg before sending.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to ffmpeg binary (default assumes it is on PATH).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only transcribe the first N entries from the manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination JSON for transcripts (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist transcripts to disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and show planned configs without calling the API.",
    )
    return parser.parse_args()


def load_manifest(json_path: Path) -> list[dict]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse manifest {json_path}: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Manifest {json_path} must contain a JSON array.")
    return payload


def limited(entries: Iterable[dict], limit: Optional[int]) -> Iterator[dict]:
    if limit is None:
        yield from entries
        return
    for idx, entry in enumerate(entries):
        if idx >= limit:
            break
        yield entry


def resolve_audio_path(raw_url: str, base_dir: Path) -> Path:
    raw_path = Path(raw_url)
    candidates = [
        base_dir / raw_path,
        base_dir / Path(str(raw_path).replace("audios", "audio", 1)),
        base_dir / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate audio file for '{raw_url}' under {base_dir}")


def convert_to_linear16(
    ffmpeg_path: str,
    source: Path,
    sample_rate: int,
    force: bool,
) -> Tuple[Path, bool]:
    """
    Ensure the audio is 16-bit PCM mono WAV at the requested sample rate.
    Returns (path_to_use, cleanup_required).
    """
    if not force and source.suffix.lower() == ".wav":
        return source, False

    tmp = Path(tempfile.mkstemp(prefix="google_stt_", suffix=".wav")[1])
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed to convert {source} -> WAV (stderr: {exc.stderr.decode(errors='ignore')[:500]})"
        ) from exc
    return tmp, True


def upload_to_gcs(
    local_path: Path,
    bucket_name: str,
    prefix: str,
) -> str:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "google-cloud-storage is required for --gcs-bucket uploads. Install with `pip install google-cloud-storage`."
        ) from exc

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    object_name = f"{prefix.rstrip('/')}/{local_path.stem}-{uuid.uuid4().hex}{local_path.suffix}"
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{object_name}"


def delete_gcs_uri(uri: str) -> None:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        return
    if not uri.startswith("gs://"):
        return
    bucket_name, _, object_path = uri[5:].partition("/")
    if not bucket_name or not object_path:
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.delete()


def transcribe_file(
    client: speech.SpeechClient,
    audio_path: Path,
    config: speech.RecognitionConfig,
    use_async: bool,
    timeout: int,
    gcs_bucket: Optional[str],
    gcs_prefix: str,
    keep_gcs: bool,
    ffmpeg_path: str,
    force_convert: bool,
) -> Tuple[str, Optional[list]]:
    cleanup_local = False
    uploaded_uri: Optional[str] = None
    wav_path: Path = audio_path
    try:
        # If using LINEAR16 encoding ensure file matches requirements
        if config.encoding == speech.RecognitionConfig.AudioEncoding.LINEAR16:
            wav_path, cleanup_local = convert_to_linear16(
                ffmpeg_path=ffmpeg_path,
                source=audio_path,
                sample_rate=config.sample_rate_hertz,
                force=force_convert or audio_path.suffix.lower() != ".wav",
            )

        if gcs_bucket:
            uploaded_uri = upload_to_gcs(wav_path, gcs_bucket, gcs_prefix)
            audio = speech.RecognitionAudio(uri=uploaded_uri)
        else:
            content = wav_path.read_bytes()
            audio = speech.RecognitionAudio(content=content)

        if use_async:
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=timeout)
        else:
            response = client.recognize(config=config, audio=audio)

        transcript_chunks = []
        word_timestamps = []
        for result in response.results:
            if not result.alternatives:
                continue
            top_alt = result.alternatives[0]
            transcript_chunks.append(top_alt.transcript)
            if top_alt.words:
                word_timestamps.extend(
                    [
                        {
                            "word": word_info.word,
                            "start": seconds_from_timedelta(word_info.start_time),
                            "end": seconds_from_timedelta(word_info.end_time),
                        }
                        for word_info in top_alt.words
                    ]
                )

        transcript_text = "\n".join(chunk.strip() for chunk in transcript_chunks if chunk.strip())
        return transcript_text, word_timestamps if word_timestamps else None
    finally:
        if uploaded_uri and not keep_gcs:
            delete_gcs_uri(uploaded_uri)
        if cleanup_local and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] Failed to parse existing output {path}: {exc}", file=sys.stderr)
        return []
    if isinstance(data, list):
        return data
    print(f"[WARN] Existing output {path} is not a JSON array; starting fresh.", file=sys.stderr)
    return []


def upsert_record(records: list[dict], new_record: dict, key: str) -> None:
    for idx, existing in enumerate(records):
        if existing.get(key) == new_record.get(key):
            records[idx] = new_record
            return
    records.append(new_record)


def to_encoding(value: str) -> speech.RecognitionConfig.AudioEncoding:
    try:
        return speech.RecognitionConfig.AudioEncoding[value.upper()]
    except KeyError as exc:
        valid = ", ".join(e.name for e in speech.RecognitionConfig.AudioEncoding if e != speech.RecognitionConfig.AudioEncoding.AUDIO_ENCODING_UNSPECIFIED)
        raise SystemExit(f"Unsupported encoding '{value}'. Valid options: {valid}") from exc


def seconds_from_timedelta(td) -> float:
    if hasattr(td, "total_seconds"):
        return float(td.total_seconds())
    seconds = getattr(td, "seconds", 0)
    nanos = getattr(td, "nanos", 0)
    return float(seconds + nanos / 1e9)


def main() -> None:
    global args
    args = parse_args()

    if args.timestamps and not args.use_async:
        print("[WARN] Word timestamps are better supported with long-running recognition.", file=sys.stderr)

    client = speech.SpeechClient()
    manifest = load_manifest(args.json)
    audio_base = args.audio_root or args.json.parent

    output_path = None if args.no_save else args.output
    output_records: list[dict] = []
    if output_path:
        output_records = load_existing_records(output_path)

    encoding_enum = to_encoding(args.encoding)

    processed = 0
    try:
        for entry in limited(manifest, args.limit):
            title = entry.get("title", "<untitled>")
            audio_url = entry.get("audio_url")
            if not audio_url:
                print(f"Skipping entry without audio_url: {title}", file=sys.stderr)
                continue

            try:
                audio_path = resolve_audio_path(audio_url, audio_base)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
                continue

            processed += 1
            print(f"[{processed}] Google STT: {title} ({audio_path})")

            config = speech.RecognitionConfig(
                encoding=encoding_enum,
                sample_rate_hertz=args.sample_rate,
                language_code=args.language,
                max_alternatives=args.max_alternatives,
                enable_automatic_punctuation=args.punctuation,
                model=args.model,
                enable_word_time_offsets=args.timestamps,
            )

            if args.dry_run:
                print("    (dry-run) Config:", config)
                print("    (dry-run) Audio path:", audio_path)
                continue

            try:
                transcript, words = transcribe_file(
                    client=client,
                    audio_path=audio_path,
                    config=config,
                    use_async=args.use_async,
                    timeout=args.timeout,
                    gcs_bucket=args.gcs_bucket,
                    gcs_prefix=args.gcs_prefix,
                    keep_gcs=args.keep_gcs,
                    ffmpeg_path=args.ffmpeg_path,
                    force_convert=args.convert,
                )
            except Exception as exc:
                print(f"[ERROR] Failed to transcribe {audio_path}: {exc}", file=sys.stderr)
                continue

            print(f"    Transcript (first 120 chars): {transcript[:120]!r}")

            if output_path:
                record = {
                    "title": title,
                    "date": entry.get("date"),
                    "audio_url": audio_url,
                    "episode_url": entry.get("episode_url"),
                    "transcript_ar": transcript,
                }
                if args.timestamps and words:
                    record["word_timestamps"] = words
                upsert_record(output_records, record, key="audio_url")
    finally:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(output_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {len(output_records)} transcript entries to {output_path}")


if __name__ == "__main__":
    main()
