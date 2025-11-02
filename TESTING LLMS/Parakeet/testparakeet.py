#!/usr/bin/env python3
"""
Batch wrapper around NVIDIA's sample CLI (python-clients/scripts/asr/transcribe_file.py).

The script:
  * Reads a manifest (default: SCRAPE/main.json) containing entries with `audio_url`.
  * Resolves each audio file under the supplied --audio-root (defaults to manifest parent).
  * Converts non-WAV sources to 16 kHz mono WAV with ffmpeg (in-place conversion optional).
  * Invokes the official `transcribe_file.py` for each clip, captures the output transcript,
    and writes results into TESTING LLMS/Parakeet/testparakeet.json (or a path you provide).

Environment variables required before running:
  PARAKEET_API_KEY     = nvapi-...
  PARAKEET_FUNCTION_ID = UUID from the Parakeet deployment

Example:
  python TESTING LLMS/Parakeet/testparakeet.py --json SCRAPE/main.json --audio-root SCRAPE --convert
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


DEFAULT_OUTPUT_PATH = Path("TESTING LLMS/Parakeet/testparakeet.json")
PARAKEET_CLI = Path("python-clients/scripts/asr/transcribe_file.py")
DEFAULT_SERVER = "grpc.nvcf.nvidia.com:443"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loop Parakeet transcription over a JSON manifest using NVIDIA's sample CLI.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("SCRAPE/main.json"),
        help="Manifest containing objects with `audio_url`. (default: SCRAPE/main.json)",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Base directory for audio files. Defaults to the manifest directory.",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Parakeet gRPC endpoint (default: {DEFAULT_SERVER}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination JSON for transcripts (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of manifest entries to process.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert input to 16 kHz mono WAV via ffmpeg before sending.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to ffmpeg executable (default assumes it is on PATH).",
    )
    parser.add_argument(
        "--cli",
        type=Path,
        default=PARAKEET_CLI,
        help=f"Path to transcribe_file.py (default: {PARAKEET_CLI}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print transcripts but do not write the JSON output file.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse manifest {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Manifest {path} must be a JSON array.")
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
    if not raw_url:
        raise FileNotFoundError("Empty audio_url")
    normalised = raw_url.replace("\\", "/")
    raw_path = Path(normalised)
    candidates = [
        base_dir / raw_path,
        base_dir / raw_path.name,
        base_dir / "audio" / raw_path.name,
        base_dir / "audios" / raw_path.name,
        Path.cwd() / raw_path,
        Path.cwd() / "audio" / raw_path.name,
        Path.cwd() / "audios" / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate audio file for '{raw_url}' (searched {base_dir})")


def convert_to_wav(
    ffmpeg_path: str,
    source: Path,
    sample_rate: int = 16000,
) -> Tuple[Path, bool]:
    """
    Convert to 16 kHz mono WAV using ffmpeg.
    Returns (converted_path, cleanup_required).
    """
    if source.suffix.lower() == ".wav":
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
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed for {source}: {stderr[:500]}") from exc
    return tmp, True


def call_parakeet_cli(
    cli_path: Path,
    server: str,
    api_key: str,
    function_id: str,
    input_path: Path,
) -> str:
    cmd = [
        sys.executable,
        str(cli_path),
        "--server",
        server,
        "--use-ssl",
        "--metadata",
        "function-id",
        function_id,
        "--metadata",
        "authorization",
        f"Bearer {api_key}",
        "--input-file",
        str(input_path),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        env=env,
    )
    transcript_lines = []
    for line in result.stdout.splitlines():
        if line.startswith("##"):
            transcript_lines.append(line.lstrip("# ").strip())
    transcript = "\n".join(transcript_lines).strip()
    if not transcript:
        transcript = result.stdout.strip()
    return transcript


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def upsert_record(records: list[dict], new_record: dict, key: str) -> None:
    for idx, existing in enumerate(records):
        if existing.get(key) == new_record.get(key):
            records[idx] = new_record
            return
    records.append(new_record)


def persist_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    api_key = os.getenv("PARAKEET_API_KEY")
    function_id = os.getenv("PARAKEET_FUNCTION_ID")
    if not api_key or not api_key.startswith("nvapi-"):
        raise SystemExit("Set PARAKEET_API_KEY environment variable (nvapi-...).")
    if not function_id:
        raise SystemExit("Set PARAKEET_FUNCTION_ID environment variable (UUID from the deployment).")

    if not args.cli.exists():
        raise SystemExit(f"Cannot find transcribe_file.py at {args.cli}. Did you clone python-clients?")

    manifest = load_manifest(args.json)
    audio_base = args.audio_root or args.json.parent

    output_records: list[dict] = [] if args.no_save else load_existing_records(args.output)

    processed = 0
    for entry in limited(manifest, args.limit):
        title = entry.get("title", "<untitled>")
        audio_url = entry.get("audio_url")
        if not audio_url:
            print(f"[SKIP] Missing audio_url for {title}", file=sys.stderr)
            continue

        try:
            audio_path = resolve_audio_path(audio_url, audio_base)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)
            continue

        processed += 1
        print(f"[{processed}] Parakeet: {title} ({audio_path})")

        wav_path = audio_path
        cleanup = False
        if args.convert:
            try:
                wav_path, cleanup = convert_to_wav(args.ffmpeg_path, audio_path)
            except RuntimeError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                continue

        try:
            transcript = call_parakeet_cli(
                cli_path=args.cli,
                server=args.server,
                api_key=api_key,
                function_id=function_id,
                input_path=wav_path,
            )
            print(f"    Transcript (first 120 chars): {transcript[:120]!r}")
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] CLI failed for {audio_path}: {exc.stderr.strip()}", file=sys.stderr)
            transcript = ""
        finally:
            if cleanup and wav_path.exists():
                try:
                    wav_path.unlink(missing_ok=True)
                except PermissionError:
                    pass

        if args.no_save:
            continue

        record = {
            "title": title,
            "date": entry.get("date"),
            "audio_url": audio_url,
            "episode_url": entry.get("episode_url"),
            "transcript_ar": transcript,
        }
        upsert_record(output_records, record, key="audio_url")
        persist_records(args.output, output_records)

    if not args.no_save:
        print(f"\nWrote {len(output_records)} transcript entries to {args.output}")


if __name__ == "__main__":
    main()
