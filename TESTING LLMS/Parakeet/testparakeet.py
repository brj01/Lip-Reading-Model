#!/usr/bin/env python3
"""
Batch transcription helper for NVIDIA Riva / Parakeet ASR (gRPC).

Mirrors the workflow used for Whisper and Munsit:
  * Reads a JSON manifest (default: SCRAPE/main.json) with `audio_url` entries
  * Resolves each audio file locally
  * Sends it to the Parakeet endpoint via the Riva gRPC client
  * Persists transcripts to TESTING LLMS/testparakeet.json by default

The script expects the `nvidia-riva-client` package to be installed and uses
metadata headers (`function-id`, `authorization`) supplied by the user or
environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import riva.client
from riva.client.proto import riva_asr_pb2, riva_audio_pb2
import requests  # noqa: F401  # imported to ensure dependency availability when sharing venv with Munsit script


DEFAULT_OUTPUT_PATH = Path(__file__).with_suffix(".json")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_LANGUAGE_CODE = "ar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files via NVIDIA Riva/Parakeet using a manifest."
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
        help="Base directory to resolve audio paths; defaults to the manifest's parent directory.",
    )
    parser.add_argument(
        "--server",
        default=os.getenv("PARAKEET_SERVER", "grpc.nvcf.nvidia.com:443"),
        help="gRPC endpoint for the Parakeet service (default: grpc.nvcf.nvidia.com:443 or PARAKEET_SERVER env).",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Disable SSL when connecting to the server (default: SSL enabled).",
    )
    parser.add_argument(
        "--function-id",
        default=os.getenv("PARAKEET_FUNCTION_ID", ""),
        help="Function ID metadata required by NVIDIA NVCF (default pulls from PARAKEET_FUNCTION_ID env).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("PARAKEET_API_KEY"),
        help="API key / bearer token. Defaults to PARAKEET_API_KEY environment variable.",
    )
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help=f"Language code passed to Riva (default: {DEFAULT_LANGUAGE_CODE}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Target sample rate in Hz (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--punctuation",
        action="store_true",
        help="Enable automatic punctuation (disabled by default).",
    )
    parser.add_argument(
        "--verbatim",
        action="store_true",
        help="Request verbatim transcripts (default: off).",
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
        "--convert",
        action="store_true",
        help="Force conversion to 16 kHz mono PCM WAV via ffmpeg before sending to Riva.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to ffmpeg binary (default assumes it is on PATH).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and show planned requests without calling the API.",
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


def convert_to_wav(
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

    tmp = Path(tempfile.mkstemp(prefix="parakeet_", suffix=".wav")[1])
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


def make_stub(server: str, use_ssl: bool, metadata: dict[str, str]) -> riva.client.ASRServiceStub:
    auth = riva.client.Auth(uri=server, use_ssl=use_ssl, metadata=metadata or None)
    return riva.client.ASRServiceStub(auth)


def recognise_file(
    stub: riva.client.ASRServiceStub,
    wav_path: Path,
    sample_rate: int,
    language_code: str,
    punctuation: bool,
    verbatim: bool,
) -> str:
    config = riva_asr_pb2.RecognitionConfig(
        encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        max_alternatives=1,
        enable_automatic_punctuation=punctuation,
        verbatim_transcripts=verbatim,
    )
    with wav_path.open("rb") as audio_stream:
        audio_content = audio_stream.read()
    request = riva_asr_pb2.RecognizeRequest(
        config=config,
        audio=riva_asr_pb2.RecognitionAudio(audio_content=audio_content),
    )
    response = stub.Recognize(request)
    transcripts = []
    for result in response.results:
        if not result.alternatives:
            continue
        transcripts.append(result.alternatives[0].transcript)
    return " ".join(transcripts).strip()


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


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Provide an API key via --api-key or PARAKEET_API_KEY environment variable.")

    metadata: dict[str, str] = {"authorization": f"Bearer {args.api_key}"}
    if args.function_id:
        metadata["function-id"] = args.function_id

    manifest = load_manifest(args.json)
    audio_base = args.audio_root or args.json.parent

    output_path = None if args.no_save else args.output
    output_records: list[dict] = []
    if output_path:
        output_records = load_existing_records(output_path)

    stub: Optional[riva.client.ASRServiceStub] = None
    if not args.dry_run:
        stub = make_stub(
            server=args.server,
            use_ssl=not args.no_ssl,
            metadata=metadata,
        )

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
            print(f"[{processed}] Parakeet: {title} ({audio_path})")

            tmp_wav: Optional[Path] = None
            cleanup = False
            try:
                wav_path, cleanup = convert_to_wav(
                    ffmpeg_path=args.ffmpeg_path,
                    source=audio_path,
                    sample_rate=args.sample_rate,
                    force=args.convert or audio_path.suffix.lower() != ".wav",
                )
                tmp_wav = wav_path if cleanup else None

                if args.dry_run:
                    print("    (dry-run) conversion target:", wav_path)
                    transcript = ""
                else:
                    assert stub is not None
                    transcript = recognise_file(
                        stub=stub,
                        wav_path=wav_path,
                        sample_rate=args.sample_rate,
                        language_code=args.language_code,
                        punctuation=args.punctuation,
                        verbatim=args.verbatim,
                    )

                if transcript:
                    print(f"    Transcript (first 120 chars): {transcript[:120]!r}")
                elif args.dry_run:
                    print("    (dry-run) Skipping transcription.")

                if output_path:
                    record = {
                        "title": title,
                        "date": entry.get("date"),
                        "audio_url": audio_url,
                        "episode_url": entry.get("episode_url"),
                        "transcript_ar": transcript,
                    }
                    upsert_record(output_records, record, key="audio_url")
            finally:
                if cleanup and tmp_wav and tmp_wav.exists():
                    tmp_wav.unlink(missing_ok=True)
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
