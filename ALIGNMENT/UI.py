#!/usr/bin/env python3
"""
MoviePy-based karaoke renderer — SCROLL-UP mode (option C).

Behavior:
- All sentences are laid out vertically.
- The "active" sentence (the one whose time range contains t) is centered vertically.
- Other sentences appear above/below it and move as the active sentence changes,
  producing a scroll-up effect with clean per-word highlighting.
- RTL shaping (Arabic/Hebrew) supported via `arabic_reshaper` + `python-bidi`.

Requirements:
- moviepy
- pillow
- numpy
- arabic-reshaper

Run:
    python karaoke_scroll.py align_myfile.json --rtl --font /path/to/arabic.ttf
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


import numpy as np
from moviepy import AudioFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display


# -------------------------
# Argument parsing
# -------------------------
def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "karaoke_scroll3.mp4"
    parser = argparse.ArgumentParser(description="Render scroll-up karaoke MP4 from align_*.json")
    parser.add_argument("alignment", type=Path, help="Path to align_*.json")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--font-size", type=int, default=80)
    parser.add_argument("--font", type=Path, default=None, help="Optional TTF/OTF font path")
    parser.add_argument("--bg-color", default="#000000")
    parser.add_argument("--text-color", default="#d9d9d9")
    parser.add_argument("--highlight-color", default="#39ff14")
    parser.add_argument("--margin", type=int, default=120)
    parser.add_argument("--line-spacing", type=int, default=18)
    parser.add_argument(
        "--rtl",
        action="store_true",
        help="Enable right-to-left shaping/alignment (use for Arabic/Hebrew text).",
    )
    parser.add_argument("--fade-duration", type=float, default=0.6, help="Fade duration (seconds) for previous/next lines")
    return parser.parse_args()


# -------------------------
# Alignment loading / fallback
# -------------------------
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
        return [{"word": tok, "start": start, "end": start} for tok in tokens]

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


# -------------------------
# Font & shaping helpers
# -------------------------
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
        if Path(candidate).exists():
            return ImageFont.truetype(str(candidate), font_size)
    raise RuntimeError("No TrueType font found. Supply --font /path/to/font.ttf")


def shape_text(text: str, rtl: bool) -> str:
    """Apply Arabic reshaping + bidi ordering for RTL scripts when requested."""
    if not rtl:
        return text
    stripped = text.strip()
    if not stripped:
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def text_size(font: ImageFont.FreeTypeFont, text: str) -> Tuple[int, int]:
    """Return (width, height) for given text with font. Uses getbbox if available for accuracy."""
    try:
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return int(w), int(h)
    except Exception:
        return font.getsize(text)


# -------------------------
# Sentence grouping & layout
# -------------------------
SENTENCE_END_PUNC = r"[.!?؟؛؛\n]"


def group_words_to_sentences(words: List[Dict]) -> List[List[Dict]]:
    """Group words into sentences. Key heuristic: punctuation tokens or long gaps between words."""
    sentences: List[List[Dict]] = []
    current: List[Dict] = []
    # Sort by start time just in case
    words_sorted = sorted(words, key=lambda w: (float(w.get("start", 0.0)), float(w.get("end", 0.0))))
    for i, w in enumerate(words_sorted):
        current.append(w)
        # end punctuation check
        token = (w.get("word") or "").strip()
        if token and re.search(SENTENCE_END_PUNC, token[-1:]):
            sentences.append(current)
            current = []
            continue
        # lookahead gap: if next word's start is much later -> split (helps with transcripts lacking punctuation)
        if i + 1 < len(words_sorted):
            next_start = float(words_sorted[i + 1].get("start") or 0.0)
            cur_end = float(w.get("end") or 0.0)
            if next_start - cur_end > 0.7:  # gap threshold (seconds) -- reasonable default
                sentences.append(current)
                current = []
    if current:
        sentences.append(current)
    # Filter out empty sentences
    return [s for s in sentences if len(s) > 0]


def layout_sentences(
    sentences: List[List[Dict]],
    font: ImageFont.FreeTypeFont,
    frame_w: int,
    margin: int,
    line_spacing: int,
    rtl: bool,
) -> List[Dict]:
    """
    For each sentence produce:
      - start (first word start), end (last word end)
      - rendered_lines: list of (rendered_text, line_width, line_height, per-word metadata with x offsets)
      - total_height (sum of line heights + spacing)
    This layout respects max width (frame_w - margin*2) and wraps long sentences.
    """
    max_width = frame_w - margin * 2
    results: List[Dict] = []

    for sent in sentences:
        words_texts = [w.get("word", "") for w in sent]
        # We'll break into tokens and build lines greedily
        tokens = []
        for w in sent:
            tok = (w.get("word") or "").strip()
            if not tok:
                continue
            tokens.append({"text": tok, "start": float(w.get("start") or 0.0), "end": float(w.get("end") or 0.0)})

        # Build lines
        lines: List[List[Dict]] = []
        current_line: List[Dict] = []
        current_w = 0
        space_w, _ = text_size(font, " ")
        for tok in tokens:
            rendered_tok = shape_text(tok["text"], rtl)
            tok_w, tok_h = text_size(font, rendered_tok)
            proposed = tok_w if not current_line else current_w + space_w + tok_w
            if current_line and proposed > max_width:
                lines.append(current_line)
                current_line = [dict(tok, render=rendered_tok, w=tok_w, h=tok_h)]
                current_w = tok_w
            else:
                current_w = tok_w if not current_line else current_w + space_w + tok_w
                current_line.append(dict(tok, render=rendered_tok, w=tok_w, h=tok_h))
        if current_line:
            lines.append(current_line)

        # For each line compute per-word x offset relative to centered line start.
        rendered_lines = []
        total_h = 0
        for line in lines:
            line_w = sum(item["w"] for item in line) + space_w * (len(line) - 1 if len(line) > 0 else 0)
            # compute x offsets
            x_offsets = []
            if rtl:
                # for RTL, words are drawn right-to-left: compute starting x (right)
                reversed_items = list(reversed(line))
                # we will subtract token widths; but to center a RTL line, we compute leftmost start as (cursor - line_w)
                cursor = frame_w - margin
                x_start = cursor - line_w
                cur_x = x_start
                for item in reversed_items:
                    x_offsets.append(cur_x)
                    cur_x += item["w"] + space_w
                x_offsets = list(reversed(x_offsets))
            else:
                x_start = (frame_w - line_w) / 2
                cur_x = x_start
                for item in line:
                    x_offsets.append(cur_x)
                    cur_x += item["w"] + space_w

            line_h = max(item["h"] for item in line) if line else int(font.size)
            rendered_lines.append({"items": line, "x_offsets": x_offsets, "line_w": line_w, "line_h": line_h})
            total_h += line_h + line_spacing
        if total_h > 0:
            total_h -= line_spacing  # last line doesn't need extra spacing

        results.append(
            {
                "start": float(sent[0].get("start") or 0.0),
                "end": float(sent[-1].get("end") or 0.0),
                "rendered_lines": rendered_lines,
                "total_height": total_h,
                "words": tokens,
            }
        )

    return results


# -------------------------
# Frame rendering factory
# -------------------------
def render_frame_factory(
    sentences_layout: List[Dict],
    frame_w: int,
    frame_h: int,
    font: ImageFont.FreeTypeFont,
    bg_color: str,
    base_color: str,
    highlight_color: str,
    margin: int,
    line_spacing: int,
    rtl: bool,
    fade_duration: float,
):
    """
    Returns a function make_frame(t) for MoviePy where:
    - active sentence is vertically centered
    - lines above/below placed relative to active sentence
    - active word is highlighted
    - older/future sentences are slightly faded (alpha) using fade_duration
    """
    # Precompute per-sentence line heights cumulative for placement
    # We'll compute y positions on the fly relative to the active sentence center.
    def find_active_sentence_index(t: float) -> Optional[int]:
        for i, s in enumerate(sentences_layout):
            if s["start"] <= t <= s["end"]:
                return i
        # If none active, pick the nearest upcoming sentence for future center,
        # or the last finished sentence if t past end.
        for i, s in enumerate(sentences_layout):
            if t < s["start"]:
                return i
        return len(sentences_layout) - 1 if sentences_layout else None

    # Utility to draw text with alpha using RGBA blending
    def draw_text_with_alpha(draw: ImageDraw.Draw, pos: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int, int]):
        # Pillow's ImageDraw doesn't support alpha in text fill unless image is RGBA.
        draw.text(pos, text, font=font, fill=fill)

    def make_frame(t: float):
        # background RGBA
        img = Image.new("RGBA", (frame_w, frame_h), bg_color)
        draw = ImageDraw.Draw(img)

        idx = find_active_sentence_index(t)
        if idx is None:
            # nothing to draw
            return np.array(img.convert("RGB"))

        active = sentences_layout[idx]
        # center y for active sentence (we'll position the sentence's block such that its center is frame_h/2)
        active_block_h = active["total_height"]
        center_y = frame_h // 2
        # start y for active block (top)
        start_y_active = center_y - (active_block_h // 2)

        # compute top y for sentence at index idx (this is baseline for that sentence)
        # For sentences above, walk backward summing total_heights + spacing
        y_positions = [0] * len(sentences_layout)
        y_positions[idx] = start_y_active
        # above
        cur_y = start_y_active
        for i in range(idx - 1, -1, -1):
            # we want sentence i to be placed above sentence i+1 with a small gap (line_spacing)
            cur_y = cur_y - (sentences_layout[i]["total_height"] + line_spacing)
            y_positions[i] = cur_y
        # below
        cur_y = start_y_active + active_block_h + line_spacing
        for i in range(idx + 1, len(sentences_layout)):
            y_positions[i] = cur_y
            cur_y = cur_y + sentences_layout[i]["total_height"] + line_spacing

        # Draw sentences
        for s_i, s in enumerate(sentences_layout):
            y_top = int(y_positions[s_i])
            block_top = y_top
            # compute fade/alpha based on temporal distance from active
            # older sentences fade out after fade_duration from their end; future sentences fade in fade_duration before start
            alpha = 255
            # If sentence finished before t:
            if t > s["end"]:
                dt = t - s["end"]
                alpha = int(max(0.0, 1.0 - (dt / fade_duration)) * 255) if fade_duration > 0 else 255
            elif t < s["start"]:
                dt = s["start"] - t
                alpha = int(max(0.0, 1.0 - (dt / fade_duration)) * 255) if fade_duration > 0 else 255
            else:
                alpha = 255

            # Use alpha to dim non-active sentences slightly
            is_active_sentence = (s_i == idx)
            word_active_index = None
            # find which word in this sentence is active (if active)
            if is_active_sentence:
                # find the active word index within s["words"]
                for wi, w in enumerate(s["words"]):
                    if w["start"] <= t <= w["end"]:
                        word_active_index = wi
                        break

            # draw lines of this sentence
            y_cursor = block_top
            for line in s["rendered_lines"]:
                items = line["items"]
                offsets = line["x_offsets"]
                line_h = line["line_h"]
                # for each word in line, its corresponding original token index is needed to match highlights.
                for j, item in enumerate(items):
                    x = int(offsets[j])
                    y = int(y_cursor)
                    text = item["render"]
                    # determine if this particular word is active:
                    # we need to map this rendered token to the corresponding token index in s["words"]
                    # Simple heuristic: match by order (the layout preserved original token order).
                    # We'll track a running counter across lines to map to tokens.
                    # To implement this mapping, compute token_index:
                    # compute how many items we've drawn in previous lines
                    # (we'll compute this once before the loop)
                    pass  # replaced below with full loop that maintains token counter
                y_cursor += line_h + line_spacing

        # Because mapping tokens to per-line items requires a running index, we'll re-render with token counting:
        img = Image.new("RGBA", (frame_w, frame_h), bg_color)
        draw = ImageDraw.Draw(img)
        for s_i, s in enumerate(sentences_layout):
            y_top = int(y_positions[s_i])
            alpha = 255
            if t > s["end"]:
                dt = t - s["end"]
                alpha = int(max(0.0, 1.0 - (dt / fade_duration)) * 255) if fade_duration > 0 else 255
            elif t < s["start"]:
                dt = s["start"] - t
                alpha = int(max(0.0, 1.0 - (dt / fade_duration)) * 255) if fade_duration > 0 else 255
            is_active_sentence = (s_i == idx)

            token_counter = 0
            y_cursor = y_top
            for line in s["rendered_lines"]:
                items = line["items"]
                offsets = line["x_offsets"]
                line_h = line["line_h"]
                for j, item in enumerate(items):
                    x = int(offsets[j])
                    y = int(y_cursor)
                    text = item["render"]
                    # We expect s["words"][token_counter] to correspond to this item.
                    # Protect index bounds:
                    if token_counter < len(s["words"]):
                        wmeta = s["words"][token_counter]
                    else:
                        wmeta = {"start": 0.0, "end": 0.0}

                    # decide color and alpha
                    # base RGBA
                    base_rgb = hex_to_rgb(base_color_hex)
                    highlight_rgb = hex_to_rgb(highlight_color_hex)
                    # default fill = base with alpha
                    if is_active_sentence and wmeta["start"] <= t <= wmeta["end"]:
                        fill = (highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 255)
                    else:
                        fill = (base_rgb[0], base_rgb[1], base_rgb[2], alpha)
                    # draw text on RGBA image
                    draw.text((x, y), text, font=font, fill=fill)
                    token_counter += 1
                y_cursor += line_h + line_spacing

        # convert to RGB for MoviePy
        return np.array(img.convert("RGB"))

    # We need colors in outer scope for conversion inside closure
    return make_frame


# -------------------------
# Utility color helpers
# -------------------------
def hex_to_rgb(hexcolor: str) -> Tuple[int, int, int]:
    h = hexcolor.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# -------------------------
# Main
# -------------------------
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

    # Group words into sentences using heuristics
    sentences = group_words_to_sentences(data["words"])
    if not sentences:
        raise ValueError("No sentences found after grouping words.")

    # Layout sentences (compute wrapping, per-line x offsets, heights)
    sentences_layout = layout_sentences(
        sentences,
        font,
        frame_w=args.width,
        margin=args.margin,
        line_spacing=args.line_spacing,
        rtl=args.rtl,
    )

    # Expose colors into closure scope (used inside make_frame)
    global base_color_hex, highlight_color_hex
    base_color_hex = args.text_color
    highlight_color_hex = args.highlight_color
    audio_path = Path("SCRAPE/audio/9903480-gibran-khalil-gibran.mp3").resolve()
    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration  # use audio duration for video

    frame_fn = render_frame_factory(
        sentences_layout,
        frame_w=args.width,
        frame_h=args.height,
        font=font,
        bg_color=args.bg_color,
        base_color=args.text_color,
        highlight_color=args.highlight_color,
        margin=args.margin,
        line_spacing=args.line_spacing,
        rtl=args.rtl,
        fade_duration=args.fade_duration,
    )

    video = VideoClip(frame_function=frame_fn, duration=duration).with_audio(audio_clip)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
    str(args.output),
    fps=24,
    codec="libx264",
    audio_codec="aac",
    audio_bitrate="192k",
    threads=2,
    preset="medium",
)

    audio_clip.close()
    video.close()


if __name__ == "__main__":
    # Keep top-level names used inside closure defined
    base_color_hex = "#d9d9d9"
    highlight_color_hex = "#39ff14"
    main()
