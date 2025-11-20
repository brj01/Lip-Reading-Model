#!/usr/bin/env python3
"""Create a cleaned timestamped JSON by aligning `mainn.json` sentences to Whisper timestamps.

This script:
  - loads the Whisper verbose JSON (or a list of per-episode Whisper outputs)
  - loads `mainn.json` (ground truth)
  - matches episodes by the filename ending of the audio_url
  - splits the ground-truth transcript into sentences
  - for each sentence, heuristically finds the closest whisper word timestamps
  - bundles sentences into chunks that do not exceed the target seconds (default 29s)
  - writes an output JSON where each episode contains audio metadata from mainn and
    a list of timestamped chunks (using Whisper timestamps as guesses)

Notes:
  - This is a heuristic: Whisper timestamps are used as estimates and `mainn.json` is
    treated as the single source of truth for text.
  - The algorithm uses difflib.SequenceMatcher on token lists to find matching spans.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import difflib
import math
import time


# Top-level parameters (adjust here, not via CLI)
# TARGET_SECONDS: preferred maximum seconds per chunk (will never be exceeded)
# TARGET_SECONDS: preferred maximum seconds per chunk (will never be exceeded)
TARGET_SECONDS: float = 28.0
# CLOSE_THRESHOLD: how close (in seconds) to TARGET_SECONDS we try to get before
# forcing a cut to avoid creating a very short tail chunk. Smaller -> stricter packing.
CLOSE_THRESHOLD: float = 3.0
# MAX_GAP_SECONDS: if the gap between consecutive sentence timestamps is larger
# than this, do not merge across the gap (flush the current chunk).
MAX_GAP_SECONDS: float = 2.0
# PAD_SECONDS: how much to extend start and end when finalizing chunks (use video timestamps padding)
PAD_SECONDS: float = 0.5
# If a sentence is very long, split it for better matching (token count)
MAX_SENT_TOKENS: int = 20
# fuzzy match settings
FUZZY_MIN_RATIO: float = 0.55
FUZZY_WINDOW_PAD: int = 6

# WER-based matching threshold: accept WER <= this (fraction of edits / ref words)
WER_MAX: float = 0.6
# When searching windows for WER matching, allow +/- this many tokens around ref len
WER_WINDOW_PAD_TOKENS: int = 8


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def basename_from_url(url: str) -> str:
    # strip backslashes and forward slashes, take last part
    return os.path.basename(url.replace('\\', '/'))


def split_sentences(text: str) -> List[str]:
    # split on newlines and sentence-ending punctuation, keep short results filtered
    if not text:
        return []
    # normalize newlines
    text = text.replace('\r', '\n')
    # split on newlines first
    parts = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # split by Arabic or Latin sentence punctuation: . ? ! ؟ … ;
        pieces = re.split(r'(?<=[\.!?\u061F\u06D4])\s+', line)
        for p in pieces:
            p = p.strip()
            if p:
                parts.append(p)
    return parts


def words(s: str) -> List[str]:
    if not s:
        return []
    return [w for w in re.split(r"\s+", s.strip()) if w]


def normalize_text(s: str) -> str:
    r"""Normalize text for matching: lower, remove punctuation, Arabic diacritics/tatweel, normalize ALEF/YEH/TEH where helpful."""
    if not s:
        return ''
    t = s.strip()
    # lower
    t = t.lower()
    # remove tatweel
    t = re.sub(r'\u0640', '', t)
    # remove Arabic diacritics and vowel marks
    t = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]', '', t)
    # normalize alef variants to bare alef
    t = re.sub(r'[\u0622\u0623\u0625]', '\u0627', t)
    # normalize final ya (alef maqsura) to ya
    t = re.sub(r'\u0649', '\u064A', t)
    # remove punctuation (keep Arabic letters and numbers and ascii letters)
    t = re.sub(r'[^\w\u0600-\u06FF\s]', ' ', t)
    # collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def build_whisper_word_list(whisper_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # payload expected to have 'words' or segments with words
    words_list: List[Dict[str, Any]] = []
    if 'words' in whisper_payload and isinstance(whisper_payload['words'], list) and len(whisper_payload['words']) > 0:
        for w in whisper_payload['words']:
            token = (w.get('word') or w.get('text') or '').strip()
            if not token:
                continue
            words_list.append({'word': token, 'norm': normalize_text(token), 'start': float(w.get('start', 0.0)), 'end': float(w.get('end', 0.0))})
    else:
        # fallback: gather words from segments
        for seg in whisper_payload.get('segments', []) or []:
            for w in seg.get('words', []) or []:
                token = (w.get('word') or w.get('text') or '').strip()
                if not token:
                    continue
                words_list.append({'word': token, 'norm': normalize_text(token), 'start': float(w.get('start', 0.0)), 'end': float(w.get('end', 0.0))})

        # If words_list is still empty, synthesize approximate per-word timestamps by
        # splitting each segment.text into tokens and distributing the segment duration
        # evenly across tokens. This helps when Whisper produced segment-level timestamps
        # but no per-word timings (common with some verbose outputs).
        if not words_list:
            segs = whisper_payload.get('segments') or []
            # sometimes verbose data is nested under 'raw' -> 'segments'
            if not segs and isinstance(whisper_payload.get('raw'), dict):
                segs = whisper_payload['raw'].get('segments') or []

            for seg in segs:
                seg_text = (seg.get('text') or '').strip()
                if not seg_text:
                    continue
                seg_start = float(seg.get('start', 0.0) or 0.0)
                seg_end = float(seg.get('end', seg_start) or seg_start)
                toks = [t for t in re.split(r"\s+", seg_text) if t]
                if not toks:
                    continue
                duration = max(0.0, seg_end - seg_start)
                per = duration / len(toks) if duration > 0 else 0.0
                for i, token in enumerate(toks):
                    wstart = seg_start + i * per if per > 0 else seg_start
                    wend = wstart + per if per > 0 else seg_end
                    words_list.append({'word': token, 'norm': normalize_text(token), 'start': float(wstart), 'end': float(wend)})
    return words_list


def map_sentence_to_timestamp(sentence: str, whisper_words: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """Heuristic: align sentence tokens to whisper word tokens and return (start,end).

    Returns None if no reasonable alignment found.
    """
    sent_tokens = words(sentence)
    # use normalized tokens for matching
    sent_norm_tokens = [normalize_text(t) for t in sent_tokens]
    whisper_tokens = [w.get('norm') or normalize_text(w.get('word', '')) for w in whisper_words]
    if not sent_tokens or not whisper_tokens:
        return None
    # SequenceMatcher on normalized tokens
    sm = difflib.SequenceMatcher(a=sent_norm_tokens, b=whisper_tokens)
    # collect matched b ranges corresponding to equal opcodes
    starts = []
    ends = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal' and j2 > j1:
            starts.append(j1)
            ends.append(j2)

    if starts:
        jstart = min(starts)
        jend = max(ends) - 1
        # clamp indices
        jstart = max(0, min(jstart, len(whisper_words)-1))
        jend = max(0, min(jend, len(whisper_words)-1))
        start_t = whisper_words[jstart]['start']
        end_t = whisper_words[jend]['end']
        return (start_t, end_t)

    # fallback: try to find first and last occurrence of a subset of normalized tokens
    # find first token occurrence
    last_found = -1
    first_idx = None
    last_idx = None
    for tok_norm in sent_norm_tokens:
        # find tok_norm in whisper_tokens after last_found
        try:
            idx = whisper_tokens.index(tok_norm, last_found+1)
        except ValueError:
            # try partial matching on normalized tokens
            found = False
            for i in range(last_found+1, len(whisper_tokens)):
                wt = whisper_tokens[i]
                if not wt:
                    continue
                if wt.startswith(tok_norm[:3]) or tok_norm.startswith(wt[:3]):
                    idx = i
                    found = True
                    break
            if not found:
                continue
        if first_idx is None:
            first_idx = idx
        last_idx = idx
        last_found = idx

    if first_idx is not None and last_idx is not None:
        start_t = whisper_words[first_idx]['start']
        end_t = whisper_words[last_idx]['end']
        return (start_t, end_t)

    return None


def fuzzy_match_sentence(sentence: str, whisper_words: List[Dict[str, Any]], min_ratio: float = FUZZY_MIN_RATIO) -> Optional[Tuple[float, float]]:
    """Attempt to find an approximate span for sentence in whisper_words using sliding windows
    on normalized joined text and SequenceMatcher ratio. Returns (start,end) in seconds.
    """
    sent_norm = normalize_text(sentence)
    if not sent_norm:
        return None

    # build list of normalized tokens and joined windows
    w_norms = [w.get('norm') or normalize_text(w.get('word', '')) for w in whisper_words]
    n = len(w_norms)
    if n == 0:
        return None

    # join tokens per window and compare
    best_ratio = 0.0
    best_span: Optional[Tuple[int, int]] = None
    sent_join = ' '.join(words(sent_norm))
    # window sizes around sentence token length
    s_tokens = words(sent_norm)
    base_len = max(1, len(s_tokens))
    for window in range(max(1, base_len - FUZZY_WINDOW_PAD), base_len + FUZZY_WINDOW_PAD + 1):
        if window <= 0:
            continue
        for i in range(0, max(1, n - window + 1)):
            j = i + window
            candidate = ' '.join(w_norms[i:j])
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(a=sent_join, b=candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (i, j-1)

    if best_ratio >= min_ratio and best_span is not None:
        i, j = best_span
        start_t = whisper_words[i]['start']
        end_t = whisper_words[j]['end']
        return (start_t, end_t)
    return None


def split_long_sentence(sentence: str, max_tokens: int = MAX_SENT_TOKENS) -> List[str]:
    """Split a long sentence into smaller pieces using punctuation boundaries where possible,
    otherwise split by token count to improve matching granularity."""
    toks = words(sentence)
    if len(toks) <= max_tokens:
        return [sentence]

    # try splitting on commas/؛/،/؛ or conjunctions via punctuation
    parts = re.split(r'[,،؛;:\-]\s*', sentence)
    parts = [p.strip() for p in parts if p.strip()]
    # if resulting pieces are still too long, further split by token count
    out: List[str] = []
    for p in parts:
        p_toks = words(p)
        if len(p_toks) <= max_tokens:
            out.append(p)
        else:
            # split into chunks of max_tokens preserving order
            for i in range(0, len(p_toks), max_tokens):
                out.append(' '.join(p_toks[i:i+max_tokens]))
    if not out:
        # fallback: split by token count on the original sentence
        for i in range(0, len(toks), max_tokens):
            out.append(' '.join(toks[i:i+max_tokens]))
    return out


def _levenshtein_word_alignment(ref_tokens: List[str], hyp_tokens: List[str]):
    """Return edit distance and alignment ops between word token lists.

    Returns (edits, ops) where ops is a list of (op, ref_tok, hyp_tok, ref_idx, hyp_idx)
    op in {'eq','sub','ins','del'}
    """
    m = len(ref_tokens)
    n = len(hyp_tokens)
    # dp table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution/equal
            )
    # backtrace
    i = m
    j = n
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            ops.append(('eq', ref_tokens[i - 1], hyp_tokens[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('sub', ref_tokens[i - 1], hyp_tokens[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('del', ref_tokens[i - 1], '', i - 1, None))
            i -= 1
        else:
            ops.append(('ins', '', hyp_tokens[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    edits = sum(1 for o in ops if o[0] != 'eq')
    return edits, ops


def compute_wer_and_alignment(ref: str, hyp: str):
    rref = normalize_text(ref)
    rhyp = normalize_text(hyp)
    ref_toks = words(rref)
    hyp_toks = words(rhyp)
    if not ref_toks:
        return 1.0, len(hyp_toks), []
    edits, ops = _levenshtein_word_alignment(ref_toks, hyp_toks)
    wer = edits / max(1, len(ref_toks))
    return wer, edits, ops


def match_by_wer(sentence: str, whisper_words: List[Dict[str, Any]], max_wer: float = WER_MAX) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """Slide windows over whisper_words and pick the window with lowest WER.
    If best WER <= max_wer, return (start_sec, end_sec, meta_dict) where meta contains 'wer' and 'diff' ops.
    """
    sent_norm = normalize_text(sentence)
    if not sent_norm:
        return None
    s_toks = words(sent_norm)
    base_len = max(1, len(s_toks))
    w_norms = [w.get('norm') or normalize_text(w.get('word', '')) for w in whisper_words]
    n = len(w_norms)
    if n == 0:
        return None
    best = (1.1, None, None, None)  # wer, i, j, ops
    min_w = max(1, base_len - WER_WINDOW_PAD_TOKENS)
    max_w = base_len + WER_WINDOW_PAD_TOKENS
    # To avoid pathological runtimes on very long whisper token lists, limit the
    # number of windows we examine. We sample start positions if needed and stop
    # early once we've done enough checks.
    max_checks = 2000
    checks = 0
    for window in range(min_w, max(1, max_w) + 1):
        max_start = max(1, n - window + 1)
        if max_start <= 0:
            continue
        # if there are too many possible starts, sample them uniformly to cap checks
        if max_start > max_checks:
            step = max(1, max_start // max_checks)
            starts = range(0, max_start, step)
        else:
            starts = range(0, max_start)
        for i in starts:
            if checks >= max_checks:
                break
            j = i + window
            candidate = ' '.join(w_norms[i:j])
            wer, edits, ops = compute_wer_and_alignment(sent_norm, candidate)
            checks += 1
            if wer < best[0]:
                best = (wer, i, j - 1, ops)
        if checks >= max_checks:
            break
    if best[0] <= max_wer and best[1] is not None:
        i = best[1]
        j = best[2]
        start_t = whisper_words[i]['start']
        end_t = whisper_words[j]['end']
        # build a mapped_text using the whisper words in the selected window
        mapped_words = [whisper_words[k].get('word', '').strip() for k in range(i, j + 1)]
        mapped_text = ' '.join(w for w in mapped_words if w)
        meta = {'approximate': True, 'wer': float(best[0]), 'edits': sum(1 for o in best[3] if o[0] != 'eq'), 'mapped_text': mapped_text}
        return (start_t, end_t, meta)
    return None


def bundle_sentences_into_chunks(sentence_timestamps: List[Tuple[str, Optional[Tuple[float, float]], Optional[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    """Bundle sequential sentences into chunks.

    Rules:
      - Preserve original sentence order.
      - Never exceed TARGET_SECONDS per chunk.
      - Prefer to cut only when a chunk is close to TARGET_SECONDS (within CLOSE_THRESHOLD).
      - Do not merge across large gaps (> MAX_GAP_SECONDS).
      - Sentences without timestamps are handled conservatively: they will be placed
        in the current chunk if they are adjacent; if isolated they may form their
        own chunk with None times.
    """
    chunks: List[Dict[str, Any]] = []
    if not sentence_timestamps:
        return chunks

    cur_sentences: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    def _finalize_chunk(s: Optional[float], e: Optional[float], sents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize a chunk: compute start/end from sentence dicts (or use s/e), apply padding,
        and return chunk with 'sentences' being the list of sentence dicts.
        """
        # build text from sentence texts
        texts = [sd.get('text', '') for sd in sents]
        text_joined = ' '.join(t for t in texts if t)

        # if no timestamps available for the chunk, return with None times
        starts = [sd['start'] for sd in sents if sd.get('start') is not None]
        ends = [sd['end'] for sd in sents if sd.get('end') is not None]
        if not starts or not ends:
            return {'start': None, 'end': None, 'text': text_joined, 'sentences': sents.copy()}

        chunk_start = min(starts) if s is None else s
        chunk_end = max(ends) if e is None else e

        # apply padding
        padded_start = max(0.0, float(chunk_start) - PAD_SECONDS)
        padded_end = float(chunk_end) + PAD_SECONDS
        return {'start': padded_start, 'end': padded_end, 'text': text_joined, 'sentences': sents.copy()}

    for sent, ts, meta in sentence_timestamps:
        # Build a sentence dict with optional timestamps and meta
        sent_dict: Dict[str, Any] = {'text': sent, 'start': None, 'end': None}
        if meta:
            # copy possible meta fields (wer, approx flag, edits, mapped_text)
            if 'wer' in meta:
                sent_dict['wer'] = float(meta['wer'])
            if 'approximate' in meta:
                sent_dict['approximate'] = bool(meta.get('approximate'))
            if 'edits' in meta:
                sent_dict['edits'] = int(meta.get('edits', 0))
            # If the WER mapping provided a mapped_text, prefer it as the sentence text
            if 'mapped_text' in meta and meta.get('mapped_text'):
                sent_dict['text'] = str(meta['mapped_text'])
        
        # Build a sentence dict with optional timestamps
        if ts is not None:
            s, e = ts
            sent_dict['start'] = float(s)
            sent_dict['end'] = float(e)

        # If a single sentence maps to a duration longer than TARGET_SECONDS,
        # split it into smaller sentence segments by token count and distribute
        # the original interval evenly across parts. Mark these as approximate.
        if sent_dict.get('start') is not None and sent_dict.get('end') is not None:
            duration = sent_dict['end'] - sent_dict['start']
            if duration > TARGET_SECONDS:
                # split into N parts
                n_parts = int(math.ceil(duration / TARGET_SECONDS))
                toks = words(sent_dict['text'])
                if not toks:
                    # cannot split by tokens, fall back to time-slices
                    part_len = duration / n_parts
                    for pi in range(n_parts):
                        part_start = sent_dict['start'] + pi * part_len
                        part_end = part_start + part_len
                        part_dict = {'text': sent_dict['text'], 'start': float(part_start), 'end': float(part_end), 'approximate': True, 'split_from_long': True}
                        # handle this part as a normal sentence (append below)
                        # we'll insert into the processing by treating as if it were the current sentence
                        # (append to cur_sentences or start new chunk as appropriate)
                        # For simplicity, push into sentence loop by converting into list
                        # We'll process these parts in-place by iterating a small list below.
                    # create parts list of token-based segments
                # token-based splitting: create exactly n_parts pieces (or fall back to time-slices)
                parts: List[Dict[str, Any]] = []
                if toks:
                    # Desired number of parts based on duration
                    # If we have enough tokens, distribute tokens evenly across n_parts
                    if len(toks) >= n_parts:
                        base = len(toks) // n_parts
                        rem = len(toks) % n_parts
                        part_duration = duration / float(n_parts)
                        idx_tok = 0
                        for i in range(n_parts):
                            this_size = base + (1 if i < rem else 0)
                            slice_tokens = toks[idx_tok: idx_tok + this_size]
                            idx_tok += this_size
                            part_start = sent_dict['start'] + i * part_duration
                            part_end = part_start + part_duration
                            part_text = ' '.join(slice_tokens)
                            parts.append({'text': part_text, 'start': float(part_start), 'end': float(part_end), 'approximate': True, 'split_from_long': True})
                    else:
                        # fewer tokens than desired parts: create time-sliced parts and assign
                        # one token to the first len(toks) parts (others will be empty text)
                        part_duration = duration / float(n_parts)
                        for i in range(n_parts):
                            slice_tokens = [toks[i]] if i < len(toks) else []
                            part_start = sent_dict['start'] + i * part_duration
                            part_end = part_start + part_duration
                            part_text = ' '.join(slice_tokens)
                            parts.append({'text': part_text, 'start': float(part_start), 'end': float(part_end), 'approximate': True, 'split_from_long': True})
                # process each part as a separate sentence dict
                for part in parts:
                    # reuse the chunking logic below by pretending this is the current sentence
                    ps = part['start']
                    pe = part['end']
                    p_sent_dict = {'text': part['text'], 'start': float(ps), 'end': float(pe)}
                    # now same logic as below for adding a timestamped sentence
                    s = p_sent_dict['start']
                    e = p_sent_dict['end']
                    if cur_start is None:
                        cur_start = s
                        cur_end = e
                        cur_sentences = [p_sent_dict]
                        continue
                    gap = s - (cur_end if cur_end is not None else s)
                    if gap > MAX_GAP_SECONDS:
                        chunks.append(_finalize_chunk(cur_start, cur_end, cur_sentences))
                        cur_start = s
                        cur_end = e
                        cur_sentences = [p_sent_dict]
                        continue
                    new_end = max(cur_end if cur_end is not None else e, e)
                    new_duration = new_end - cur_start if cur_start is not None else 0.0
                    if new_duration > TARGET_SECONDS:
                        chunks.append(_finalize_chunk(cur_start, cur_end, cur_sentences))
                        cur_start = s
                        cur_end = e
                        cur_sentences = [p_sent_dict]
                        continue
                    cur_sentences.append(p_sent_dict)
                    cur_end = new_end
                    cur_duration = (cur_end - cur_start) if cur_start is not None else 0.0
                    if cur_duration >= (TARGET_SECONDS - CLOSE_THRESHOLD):
                        chunks.append(_finalize_chunk(cur_start, cur_end, cur_sentences))
                        cur_start = None
                        cur_end = None
                        cur_sentences = []
                # done processing parts; move to next sentence
                continue

        # If sentence has no timestamp, append conservatively to current chunk if present,
        # otherwise create a small standalone chunk (no times).
        if sent_dict['start'] is None:
            if cur_start is None:
                # standalone chunk with no time
                chunks.append(_finalize_chunk(None, None, [sent_dict]))
            else:
                cur_sentences.append(sent_dict)
            continue

        s = sent_dict['start']
        e = sent_dict['end']
        if cur_start is None:
            # start new chunk
            cur_start = s
            cur_end = e
            cur_sentences = [sent_dict]
            continue

        # check gap between the previous sentence end and this sentence start
        gap = s - (cur_end if cur_end is not None else s)
        if gap > MAX_GAP_SECONDS:
            # gap too large: flush current chunk and start a new one
            chunks.append(_finalize_chunk(cur_start, cur_end, cur_sentences))
            cur_start = s
            cur_end = e
            cur_sentences = [sent_dict]
            continue

        # if adding this sentence would exceed TARGET_SECONDS, decide whether to flush now
        new_end = max(cur_end if cur_end is not None else e, e)
        new_duration = new_end - cur_start if cur_start is not None else 0.0

        if new_duration > TARGET_SECONDS:
            # cannot add this sentence to current chunk because it would exceed limit.
            chunks.append(_finalize_chunk(cur_start, cur_end, cur_sentences))
            cur_start = s
            cur_end = e
            cur_sentences = [sent_dict]
            continue

        # otherwise adding sentence keeps us within TARGET_SECONDS: append
        cur_sentences.append(sent_dict)
        cur_end = new_end

        # If current chunk has reached the close threshold, flush it now to avoid
        # creating a short tail chunk later.
        cur_duration = (cur_end - cur_start) if cur_start is not None else 0.0
        if cur_duration >= (TARGET_SECONDS - CLOSE_THRESHOLD):
            chunks.append(_finalize_chunk(cur_start, cur_end, cur_sentences))
            cur_start = None
            cur_end = None
            cur_sentences = []

    # flush remainder
    if cur_sentences:
        chunks.append(_finalize_chunk(cur_start, cur_end, cur_sentences))
    return chunks


def main():
    parser = argparse.ArgumentParser(description='Create cleaned timestamped JSON using mainn and Whisper timestamps')
    parser.add_argument('--whisper', type=Path, required=True, help='Path to whisper JSON (list or single payload)')
    parser.add_argument('--mainn', type=Path, required=True, help='Path to mainn.json ground truth')
    parser.add_argument('--out', type=Path, default=Path('cleaned_timestamped.json'), help='Output JSON path')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of whisper items to process (0 = no limit)')
    parser.add_argument('--verbose', action='store_true', help='Print progress information')
    args = parser.parse_args()

    whisper_data = load_json(args.whisper)
    # whisper_data may be either a dict (single episode) or list of dicts
    whisper_items = whisper_data if isinstance(whisper_data, list) else [whisper_data]

    mainn = load_json(args.mainn)
    # build mainn map by basename of audio_url
    mainn_map = {}
    for item in mainn:
        url = item.get('audio_url') or item.get('audio') or ''
        key = basename_from_url(url)
        if key:
            mainn_map[key] = item

    results = []
    unmatched_whisper = []

    start_run = time.time()
    total = len(whisper_items)
    for idx, w in enumerate(whisper_items):
        if args.limit and idx >= args.limit:
            if args.verbose:
                print(f'Reached --limit {args.limit}, stopping early')
            break
        if args.verbose and idx % 10 == 0:
            print(f'Processing item {idx+1}/{total} (elapsed {time.time()-start_run:.1f}s)')
        # current whisper item
        # allow idx and verbose in debug prints
        # determine audio basename: try known fields
        w_audio = w.get('audio_file') or w.get('audio_url') or ''
        w_key = basename_from_url(w_audio)
        if not w_key:
            # try to guess from any contained metadata
            w_key = ''
        main_item = mainn_map.get(w_key)
        if args.verbose:
            print(f'  whisper key={w_key!r} matched_main={bool(main_item)}')
        if not main_item:
            # couldn't match
            unmatched_whisper.append(w_key or w_audio)
            continue

        title = main_item.get('title')
        audio_url = main_item.get('audio_url') or main_item.get('audio') or w_audio
        truth_text = main_item.get('transcribed_ar') or main_item.get('transcript_ar') or ''

        # build whisper word list
        whisper_words = build_whisper_word_list(w)
        if args.verbose:
            print(f'    whisper_words={len(whisper_words)} tokens')

        # split truth into sentences
        sentences = split_sentences(truth_text)

        # map each sentence to timestamps (store optional meta for approximate matches)
        sentence_timestamps: List[Tuple[str, Optional[Tuple[float, float]], Optional[Dict[str, Any]]]] = []
        # split long sentences into smaller pieces for better alignment
        for sent in sentences:
            subs = split_long_sentence(sent)
            for sub in subs:
                ts = map_sentence_to_timestamp(sub, whisper_words)
                meta: Optional[Dict[str, Any]] = None

                # If exact mapping produced a span but it's suspiciously long relative
                # to the sentence token count, try a more permissive WER-based match
                # which is allowed to substitute words (we accept WER-based alignment
                # as the authoritative mapping when exact matches are sparse).
                if ts is not None:
                    try:
                        s_try, e_try = float(ts[0]), float(ts[1])
                    except Exception:
                        s_try, e_try = None, None
                    toks = words(sub)
                    if s_try is not None and e_try is not None and toks:
                        duration = e_try - s_try
                        # if more than 3s per token, mapping is likely too broad — try WER
                        if duration / max(1, len(toks)) > 3.0:
                            wer_match = match_by_wer(sub, whisper_words, max_wer=0.9)
                            if wer_match is not None:
                                s_t, e_t, meta = wer_match
                                ts = (s_t, e_t)

                # If still no timestamp, try fuzzy matching
                if ts is None:
                    ts = fuzzy_match_sentence(sub, whisper_words)

                # Final fallback: allow a more permissive WER-based mapping
                # (we permit high WER for very short or noisy sentences)
                if ts is None:
                    # dynamic max_wer: allow higher WER for very short sentences
                    ref_toks = words(normalize_text(sub))
                    if len(ref_toks) <= 5:
                        max_wer_allowed = 0.9
                    elif len(ref_toks) <= 10:
                        max_wer_allowed = 0.8
                    else:
                        max_wer_allowed = WER_MAX

                    wer_match = match_by_wer(sub, whisper_words, max_wer=max_wer_allowed)
                    if wer_match is not None:
                        s_t, e_t, meta = wer_match
                        ts = (s_t, e_t)

                sentence_timestamps.append((sub, ts, meta))

        # bundle into chunks not exceeding TARGET_SECONDS
        chunks = bundle_sentences_into_chunks(sentence_timestamps)

        results.append({'title': title, 'audio_url': audio_url, 'chunks': chunks})
        if args.verbose:
            print(f'  -> appended episode: {title!r}, chunks={len(chunks)} (total results so far: {len(results)})')

    out = {'results': results, 'unmatched_whisper': unmatched_whisper}
    # Write to a temp file and atomically replace the target to avoid partial writes
    tmp_path = args.out.parent / (args.out.name + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    # atomic replace (works on Windows via os.replace)
    os.replace(str(tmp_path), str(args.out))

    print(f'Wrote {len(results)} episodes to {args.out}. Unmatched whisper items: {len(unmatched_whisper)}')


if __name__ == '__main__':
    main()
