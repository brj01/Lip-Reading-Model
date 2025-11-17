#!/usr/bin/env python3
"""
Minimal alignment viewer for JSON files produced by transcribe_whisper1.py
or transcribe_gpt4o.py. Loads transcript + timestamps and renders metadata,
audio preview, and SRT/code helpers.
"""

import html
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import json


def read_alignment(json_file: str) -> Tuple[dict, Path]:
    path = Path(json_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, path


def resolve_audio(json_path: Path, audio_hint: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not audio_hint:
        return None, "No audio path found in JSON."

    hint_path = Path(audio_hint)
    candidates = [hint_path]
    if not hint_path.is_absolute():
        candidates.append(json_path.parent / hint_path)

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return str(candidate), None

    return str(Path(audio_hint)), f"Audio file not found: {audio_hint}"


def seconds_to_srt(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    millis = int(round((value - int(value)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def segments_to_srt(segments: List[dict]) -> str:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = seconds_to_srt(float(seg.get("start", 0)))
        end = seconds_to_srt(float(seg.get("end", 0)))
        text = (seg.get("text") or "").replace("\n", " ").strip()
        lines.extend([str(idx), f"{start} --> {end}", text, ""])
    return "\n".join(lines).strip()


def transcript_to_html(text: str) -> str:
    if not text:
        return "<i>No transcript available.</i>"
    return "<br>".join(html.escape(text).splitlines())


def format_metadata(data: dict, json_path: Path, audio_path: Optional[str], warning: Optional[str]) -> str:
    lines = [
        f"**Title:** {data.get('title') or json_path.stem}",
        f"**Date:** {data.get('date') or 'N/A'}",
        f"**Alignment JSON:** `{json_path}`",
        f"**Audio path:** `{data.get('audio_file') or 'N/A'}`",
    ]
    if audio_path:
        lines.append(f"**Resolved audio:** `{audio_path}`")
    if warning:
        lines.append(f"⚠️ {warning}")
    return "\n".join(lines)


def build_ffmpeg_commands(audio_source: Optional[str]) -> str:
    audio_ref = audio_source or "sample01.wav"
    return textwrap.dedent(
        f"""
        # Static image + audio -> MP4
        ffmpeg -loop 1 -i background.png -i "{audio_ref}" \\
          -c:v libx264 -tune stillimage -c:a aac -b:a 192k \\
          -pix_fmt yuv420p -shortest sample01.mp4

        # Burn SRT subtitles
        ffmpeg -loop 1 -i background.png -i "{audio_ref}" \\
          -vf "subtitles=transcript.srt:force_style='FontSize=30,PrimaryColour=&HFFFFFF&'" \\
          -c:v libx264 -c:a aac -shortest -pix_fmt yuv420p sample01.mp4
        """
    ).strip()


def build_moviepy_snippet(audio_source: Optional[str]) -> str:
    audio_literal = json.dumps(audio_source or "sample01.wav")
    return textwrap.dedent(
        f"""
        from moviepy.editor import AudioFileClip, ImageClip

        audio = AudioFileClip({audio_literal})
        bg = ImageClip("background.png").set_duration(audio.duration).resize(height=720)
        video = bg.set_audio(audio)
        video.write_videofile("sample01.mp4", fps=24, codec="libx264", audio_codec="aac")
        """
    ).strip()


def preview_alignment(json_file: Optional[str]):
    if not json_file:
        return (
            "Upload an alignment JSON.",
            None,
            "<i>Transcript preview will appear here.</i>",
            [],
            "No segments available.",
            "",
            "",
        )

    try:
        data, json_path = read_alignment(json_file)
    except Exception as exc:
        return f"Failed to read JSON: {exc}", None, "", [], "", "", ""

    transcript_text = data.get("transcript") or "\n".join(seg.get("text", "") for seg in data.get("segments", []))
    transcript_html = transcript_to_html(transcript_text)

    words = data.get("words", [])
    word_table = [
        [w.get("word", ""), round(float(w.get("start", 0)), 3), round(float(w.get("end", 0)), 3)]
        for w in words
    ]

    segments = data.get("segments", [])
    srt_preview = segments_to_srt(segments) if segments else "No segments available."

    audio_source, warning = resolve_audio(json_path, data.get("audio_file"))
    metadata = format_metadata(data, json_path, audio_source, warning)
    ffmpeg_cmds = build_ffmpeg_commands(audio_source)
    moviepy_code = build_moviepy_snippet(audio_source)
    audio_value = audio_source if audio_source and Path(audio_source).exists() else None

    return metadata, audio_value, transcript_html, word_table, srt_preview, ffmpeg_cmds, moviepy_code


with gr.Blocks(title="Alignment (OpenAI JSON)") as app:
    gr.Markdown(
        """
        ## Alignment viewer for OpenAI transcripts
        Load `align_*.json` files from `transcribe_whisper1.py` or `transcribe_gpt4o.py`
        and preview transcripts, timestamps, and helper commands.
        """
    )

    file_input = gr.File(label="Alignment JSON", file_types=[".json"], type="filepath")
    metadata_box = gr.Markdown(label="Metadata")
    audio_preview = gr.Audio(label="Audio preview", type="filepath")
    transcript_html = gr.HTML(label="Transcript")
    words_df = gr.Dataframe(headers=["word", "start", "end"], label="Word timestamps")
    srt_box = gr.Textbox(label="SRT preview", lines=10)
    ffmpeg_code = gr.Code(label="FFmpeg snippet", language="bash")
    moviepy_code = gr.Code(label="MoviePy snippet", language="python")

    file_input.change(
        fn=preview_alignment,
        inputs=file_input,
        outputs=[metadata_box, audio_preview, transcript_html, words_df, srt_box, ffmpeg_code, moviepy_code],
    )

app.launch()
