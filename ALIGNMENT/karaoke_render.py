#!/usr/bin/env python3
"""
MoviePy-based karaoke renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
from moviepy import AudioFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "karaoke1.mp4"
    parser = argparse.ArgumentParser(description="Render karaoke MP4 from align_*.json")
    parser.add_argument("alignment", type=Path, help="Path to align_*.json")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--font-size", type=int, default=96)
    parser.add_argument("--font", type=Path, default=None, help="Optional TTF/OTF font path")
    parser.add_argument("--bg-color", default="#000000")
    parser.add_argument("--text-color", default="#d9d9d9")
    parser.add_argument("--highlight-color", default="#39ff14")
    parser.add_argument("--margin", type=int, default=160)
    parser.add_argument("--line-spacing", type=int, default=30)
    parser.add_argument(
        "--rtl",
        action="store_true",
        help="Enable right-to-left shaping/alignment (use for Arabic/Hebrew text).",
    )
    return parser.parse_args()


def load_alignment(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("words"):
        segments = data.get("segments") or []
        words: List[Dict] = []
        for seg in segments:
            seg_words = seg.get("words") or []
            if seg_words:
                for word in seg_words:
                    token = (word.get("word") or word.get("text") or "").strip()
                    if not token:
                        continue
                    words.append(
                        {
                            "word": token,
                            "start": float(word.get("start") or 0.0),
                            "end": float(word.get("end") or 0.0),
                        }
                    )
            else:
                words.extend(_approximate_segment_words(seg))
        data["words"] = words
    if not data.get("words"):
        raise ValueError("Alignment JSON lacks usable word-level timestamps.")
    return data


def _approximate_segment_words(segment: Dict) -> List[Dict]:
    """Fallback: distribute segment duration evenly between whitespace tokens."""
    text = (segment.get("text") or "").strip()
    if not text:
        return []
    tokens = [tok for tok in re.split(r"\s+", text) if tok]
    if not tokens:
        return []

    start = float(segment.get("start") or 0.0)
    end = float(segment.get("end") or start)
    duration = max(end - start, 0.0)
    if duration == 0.0:
        return [
            {
                "word": tok,
                "start": start,
                "end": start,
            }
            for tok in tokens
        ]

    step = duration / len(tokens)
    words: List[Dict] = []
    current = start
    for idx, tok in enumerate(tokens):
        token_end = end if idx == len(tokens) - 1 else current + step
        words.append({"word": tok, "start": current, "end": token_end})
        current = token_end
    return words


def resolve_audio(json_path: Path, data: Dict) -> Path:
    audio_hint = (data.get("audio_file") or data.get("audio_url") or "").strip()
    if not audio_hint:
        raise ValueError("Alignment JSON missing 'audio_file' or 'audio_url'.")
    parsed = urlparse(audio_hint)
    if parsed.scheme in {"http", "https"}:
        raise ValueError("Remote audio URLs not supported; download locally.")
    candidate = Path(audio_hint)

    if candidate.is_absolute():
        audio_path = candidate
    else:
        win_hint = PureWindowsPath(audio_hint)
        if win_hint.drive:
            drive = win_hint.drive.rstrip(":").lower()
            audio_path = Path("/mnt") / drive / Path(*win_hint.parts[1:])
        else:
            audio_path = json_path.parent / candidate

    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def load_font(font_path: Optional[Path], font_size: int) -> ImageFont.FreeTypeFont:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            Path("arial.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)
    raise RuntimeError("No TrueType font found. Supply --font /path/to/font.ttf")


def shape_text(text: str, rtl: bool) -> str:
    if not rtl:
        return text
    stripped = text.strip()
    if not stripped:
        return text
    return arabic_reshaper.reshape(text)


def compute_positions(
    words: List[Dict],
    font: ImageFont.FreeTypeFont,
    frame_w: int,
    frame_h: int,
    margin: int,
    line_spacing: int,
    rtl: bool,
) -> List[Dict]:
    max_width = frame_w - margin * 2
    space_w = font.getlength(" ")
    lines: List[List[Dict]] = []
    widths: List[float] = []
    current_line: List[Dict] = []
    current_width = 0.0

    for word in words:
        text = (word.get("word") or "").strip()
        rendered = shape_text(text, rtl)
        word_w = font.getlength(rendered) if rendered else 0.0
        proposed = word_w if not current_line else current_width + space_w + word_w
        if current_line and proposed > max_width:
            lines.append(current_line)
            widths.append(current_width)
            current_line = [dict(word, render=rendered)]
            current_width = word_w
        else:
            current_width = word_w if not current_line else current_width + space_w + word_w
            current_line.append(dict(word, render=rendered))
    if current_line:
        lines.append(current_line)
        widths.append(current_width)

    total_h = len(lines) * font.size + max(0, len(lines) - 1) * line_spacing
    start_y = max((frame_h - total_h) // 2, margin)

    positioned: List[Dict] = []
    y = start_y
    for line, line_w in zip(lines, widths):
        if rtl:
            cursor = frame_w - margin
            for word in line:
                rendered = word.get("render") or shape_text(word.get("word") or "", rtl)
                word_w = font.getlength(rendered) if rendered else 0.0
                cursor -= word_w
                positioned.append(
                    {
                        "word": word.get("word") or "",
                        "render": rendered,
                        "start": float(word.get("start") or 0.0),
                        "end": float(word.get("end") or 0.0),
                        "x": cursor,
                        "y": y,
                    }
                )
                cursor -= space_w
        else:
            x = (frame_w - line_w) / 2
            for word in line:
                rendered = word.get("render") or shape_text(word.get("word") or "", rtl)
                word_w = font.getlength(rendered) if rendered else 0.0
                positioned.append(
                    {
                        "word": word.get("word") or "",
                        "render": rendered,
                        "start": float(word.get("start") or 0.0),
                        "end": float(word.get("end") or 0.0),
                        "x": x,
                        "y": y,
                    }
                )
                x += word_w + space_w
        y += font.size + line_spacing
    return positioned


def render_frame_factory(
    positioned_words: List[Dict],
    frame_w: int,
    frame_h: int,
    font: ImageFont.FreeTypeFont,
    bg_color: str,
    base_color: str,
    highlight_color: str,
):
    base_img = Image.new("RGB", (frame_w, frame_h), bg_color)
    base_draw = ImageDraw.Draw(base_img)
    for word in positioned_words:
        base_draw.text((word["x"], word["y"]), word.get("render") or word["word"], font=font, fill=base_color)

    def make_frame(t: float):
        img = base_img.copy()
        draw = ImageDraw.Draw(img)
        for word in positioned_words:
            if word["start"] <= t <= word["end"]:
                draw.text((word["x"], word["y"]), word.get("render") or word["word"], font=font, fill=highlight_color)
        return np.array(img)

    return make_frame


def main():
    args = parse_args()
    alignment_path = args.alignment.resolve()
    data = load_alignment(alignment_path)
    audio_path = resolve_audio(alignment_path, data)

    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration
    if math.isclose(duration, 0.0):
        raise ValueError("Audio duration is zero.")

    font = load_font(args.font, args.font_size)
    positioned_words = compute_positions(
        data["words"],
        font,
        frame_w=args.width,
        frame_h=args.height,
        margin=args.margin,
        line_spacing=args.line_spacing,
        rtl=args.rtl,
    )

    frame_fn = render_frame_factory(
        positioned_words,
        frame_w=args.width,
        frame_h=args.height,
        font=font,
        bg_color=args.bg_color,
        base_color=args.text_color,
        highlight_color=args.highlight_color,
    )

    video = VideoClip(frame_function=frame_fn, duration=duration).with_audio(audio_clip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(args.output),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        preset="medium",
    )
    audio_clip.close()
    video.close()


if __name__ == "__main__":
    main()
