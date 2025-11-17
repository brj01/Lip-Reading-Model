#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AraDiaWER-style evaluation pipeline (edited to use provided transcripts only).

Usage:
  python aradawer_pipeline.py --json SCRAPE/main.json --ref_dir REF_TXTS --pred_dir PRED_TXTS --out_dir results

Requirements (recommended):
  pip install jiwer sentence-transformers rapidfuzz umap-learn scikit-learn scipy pandas matplotlib transformers torch sentencepiece
  pip install camel-tools    # optional but recommended for Arabic morph tagging
"""

import os
import json
import argparse
import re
import math
from pathlib import Path
from datetime import datetime
import logging

# optional imports with graceful fallback
try:
    from camel_tools.tokenizers.word import simple_word_tokenize as camel_tokenize
    from camel_tools.taggers.morph import MorphTagger
    CAMEL_TOOLS_AVAILABLE = True
except Exception:
    CAMEL_TOOLS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

try:
    from rapidfuzz import fuzz, distance as rf_distance
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

try:
    from jiwer import compute_measures
    JIWER_AVAILABLE = True
except Exception:
    JIWER_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------------------
# Text normalization & tokenization
# ---------------------------
ARABIC_DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]+'  # diacritics & tashkeel ranges
)

def normalize_arabic(text: str) -> str:
    """Basic Arabic normalization (drop diacritics, normalize alef/ya/taa marbuta, remove punctuation)."""
    if not text:
        return ""
    text = text.strip()
    # Remove Arabic diacritics/tashkeel
    text = ARABIC_DIACRITICS_RE.sub("", text)
    # Normalize Alef forms to bare alef
    text = re.sub('[إأآا]', 'ا', text)
    # Normalize Ya
    text = re.sub('[يى]', 'ي', text)
    # Ta marbuta -> ه (common normalization choice; adjust if you prefer 'ة' kept)
    text = re.sub('ة', 'ه', text)
    # Remove punctuation (Arabic and Latin punctuation)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_arabic(text: str):
    """Tokenize Arabic text. Use CAMeL Tools if available; fallback to whitespace split."""
    if CAMEL_TOOLS_AVAILABLE:
        return camel_tokenize(text)
    else:
        return text.split()


# ---------------------------
# WER counts
# ---------------------------
def wer_counts(reference: str, hypothesis: str):
    """Return SUB, DEL, INS, REF_LEN using jiwer or simple diff fallback."""
    if JIWER_AVAILABLE:
        measures = compute_measures(reference, hypothesis)
        sub = measures.get('substitutions', 0)
        deletions = measures.get('deletions', 0)
        insertions = measures.get('insertions', 0)
        ref_len = measures.get('reference_length', 0)
        return sub, deletions, insertions, ref_len
    else:
        # very simple fallback: map to tokens and use edit distance dynamic programming
        ref_toks = reference.split()
        hyp_toks = hypothesis.split()
        n = len(ref_toks)
        m = len(hyp_toks)
        # DP table
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if ref_toks[i-1] == hyp_toks[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1,       # deletion
                               dp[i][j-1] + 1,       # insertion
                               dp[i-1][j-1] + cost)  # substitution
        # Now backtrack to count sub/del/ins
        i, j = n, m
        subs = dels = ins = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] and ref_toks[i-1] == hyp_toks[j-1]:
                i -= 1; j -= 1  # hit
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                subs += 1; i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                dels += 1; i -= 1
            else:
                ins += 1; j -= 1
        return subs, dels, ins, n


# ---------------------------
# Semantic scoring (embeddings)
# ---------------------------
def make_embedding_model(name="sentence-transformers/LaBSE"):
    """
    Default: sentence-transformers/LaBSE (language-agnostic BERT Sentence Embedding).
    LaBSE is a robust multilingual choice with good Arabic coverage; you can override with --embed_model.
    """
    if not ST_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    return SentenceTransformer(name)

def semantic_similarity_score(model, ref_text: str, hyp_text: str):
    """Return cosine similarity in [-1, 1]. Works on normalized Arabic text."""
    # ensure inputs are strings
    ref_text = ref_text or ""
    hyp_text = hyp_text or ""
    emb_ref = model.encode(ref_text, convert_to_numpy=True)
    emb_hyp = model.encode(hyp_text, convert_to_numpy=True)
    if SKLEARN_AVAILABLE:
        sim = cosine_similarity([emb_ref], [emb_hyp])[0][0]
    else:
        sim = float(np.dot(emb_ref, emb_hyp) / (np.linalg.norm(emb_ref)*np.linalg.norm(emb_hyp) + 1e-12))
    return float(sim)


# ---------------------------
# Syntactic & Linguistic scoring
# ---------------------------
def prepare_morph_tagger():
    """Return a CAMeL MorphTagger if available."""
    if CAMEL_TOOLS_AVAILABLE:
        try:
            tagger = MorphTagger.pretrained()
            return tagger
        except Exception as e:
            logging.warning("Failed to load CAMeL MorphTagger: %s", e)
            return None
    return None

def syntactic_score(tagger, ref_tokens, hyp_tokens):
    """
    If tagger available, compare morphological tags; otherwise fallback to normalized Levenshtein distance
    between token sequences (higher -> more mismatch).
    Returns a score where higher means worse syntactic match (more errors).
    """
    if tagger is not None:
        try:
            tags_ref = [m[1] for m in tagger.tag(ref_tokens)]
            tags_hyp = [m[1] for m in tagger.tag(hyp_tokens)]
            L = max(len(tags_ref), len(tags_hyp))
            mismatches = sum(1 for i in range(L) if (i >= len(tags_ref) or i >= len(tags_hyp) or tags_ref[i] != tags_hyp[i]))
            return mismatches / max(L, 1)
        except Exception as e:
            logging.warning("MorphTagger failed at syntactic scoring: %s", e)
    # fallback: normalized token-level edit distance (0..1) where 1 means completely different
    ref_join = " ".join(ref_tokens)
    hyp_join = " ".join(hyp_tokens)
    if RAPIDFUZZ_AVAILABLE:
        d = rf_distance.Levenshtein.normalized_distance(ref_join, hyp_join)
        return float(d)
    else:
        a = ref_join
        b = hyp_join
        n = len(a); m = len(b)
        if n == 0 and m == 0:
            return 0.0
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if a[i-1]==b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        return dp[n][m] / max(n, m, 1)

def linguistic_fuzzy_score(ref_tokens, hyp_tokens):
    """Apply fuzzy token-list matching (e.g., token_sort_ratio) on POS/LEX strings or tokens.
       Returns a 0..100 score (higher = more similar)."""
    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.token_sort_ratio(" ".join(ref_tokens), " ".join(hyp_tokens)))
    else:
        set_ref = set(ref_tokens)
        set_hyp = set(hyp_tokens)
        if not set_ref:
            return 0.0
        overlap = len(set_ref.intersection(set_hyp))
        return float(100.0 * overlap / len(set_ref))


# ---------------------------
# Perplexity (causal LM)
# ---------------------------
def load_causal_lm(model_name="aubmindlab/aragpt2-base", device=None):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers or torch not available. pip install transformers torch")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device

def compute_perplexity_lm(model, tokenizer, device, text, max_length=1024):
    """Compute perplexity for text using a causal LM. If text is long, split into chunks."""
    if not text:
        return float("inf")
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    stride = max_length
    n = input_ids.size(0)
    total_loss = 0.0
    total_length = 0
    for i in range(0, n, stride):
        begin_loc = i
        end_loc = min(i + stride, n)
        chunk_ids = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(chunk_ids, labels=chunk_ids)
            loss = outputs.loss.item()
        seq_len = end_loc - begin_loc
        total_loss += loss * seq_len
        total_length += seq_len
    if total_length == 0:
        return float("inf")
    avg_loss = total_loss / total_length
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    return float(ppl)


# ---------------------------
# UMAP projection helper
# ---------------------------
def make_umap_plot(embeddings, labels, outpath):
    if not UMAP_AVAILABLE:
        logging.warning("UMAP not installed, skipping UMAP plot.")
        return
    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    # color by labels (0 = ref, 1 = hyp)
    colors = ['#1f77b4' if lab==0 else '#ff7f0e' for lab in labels]
    plt.scatter(proj[:,0], proj[:,1], c=colors, alpha=0.6, s=10)
    plt.title("UMAP projection (ref vs hyp)")
    plt.savefig(outpath, dpi=150)
    plt.close()


# ---------------------------
# Main pipeline
# ---------------------------
def process_all(args):
    # load JSON episodes
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    episodes = json.loads(json_path.read_text(encoding="utf-8"))

    # load models optionally
    embedding_model = None
    if args.embed_model:
        logging.info("Loading embedding model %s ...", args.embed_model)
        embedding_model = make_embedding_model(args.embed_model)

    tagger = prepare_morph_tagger() if args.use_camel else None

    lm_model = None; lm_tokenizer = None; lm_device = None
    if args.lm_model:
        logging.info("Loading causal LM %s ...", args.lm_model)
        lm_model, lm_tokenizer, lm_device = load_causal_lm(args.lm_model)

    results = []
    all_embeddings = []
    all_labels = []  # for UMAP: 0 = ref, 1 = hyp (so we can color)
    for idx, ep in enumerate(episodes):
        title = ep.get("title", f"episode_{idx+1}")
        logging.info("[%d/%d] Processing %s", idx+1, len(episodes), title)

        # Read prediction transcript
        pred_file = Path(args.pred_dir) / f"{title}.txt"
        if not pred_file.exists():
            logging.warning("Prediction file missing for %s at %s. Skipping.", title, pred_file)
            continue
        hypothesis_text = pred_file.read_text(encoding="utf-8")

        # Read reference transcript
        ref_file = Path(args.ref_dir) / f"{title}.txt"
        if not ref_file.exists():
            logging.warning("Reference transcript missing for %s at %s. Skipping.", title, ref_file)
            continue
        reference_text = ref_file.read_text(encoding="utf-8")

        # Normalize both (use Arabic normalization)
        ref_norm = normalize_arabic(reference_text)
        hyp_norm = normalize_arabic(hypothesis_text)

        # Tokenize
        ref_tokens = tokenize_arabic(ref_norm)
        hyp_tokens = tokenize_arabic(hyp_norm)

        # WER counts
        sub, deletions, insertions, ref_len = wer_counts(" ".join(ref_tokens), " ".join(hyp_tokens))
        wer_value = (sub + deletions + insertions) / max(1, ref_len)

        # Semantic similarity (use normalized Arabic)
        sem_score = None
        if embedding_model is not None:
            sem_score = semantic_similarity_score(embedding_model, ref_norm, hyp_norm)
        else:
            sem_score = 0.0

        # Linguistic fuzzy score (0..100)
        ling_score = linguistic_fuzzy_score(ref_tokens, hyp_tokens)

        # Syntactic score (higher = worse)
        syn_score = syntactic_score(tagger, ref_tokens, hyp_tokens)

        # Perplexity (if LM available)
        ppl = None
        if lm_model is not None:
            ppl = compute_perplexity_lm(lm_model, lm_tokenizer, lm_device, hyp_norm)
        else:
            ppl = float("nan")

        # Embeddings for UMAP if available
        if embedding_model is not None:
            emb_ref = embedding_model.encode(ref_norm, convert_to_numpy=True)
            emb_hyp = embedding_model.encode(hyp_norm, convert_to_numpy=True)
            all_embeddings.append(emb_ref); all_labels.append(0)
            all_embeddings.append(emb_hyp); all_labels.append(1)

        rec = {
            "title": title,
            "audio_path": ep.get("audio_url", ""),   # optional, kept for convenience
            "ref_len": int(ref_len),
            "WER": float(wer_value),
            "SUB": int(sub),
            "DEL": int(deletions),
            "INS": int(insertions),
            "SEM_SIM": float(sem_score),
            "LING_FUZZY": float(ling_score),
            "SYN_SCORE": float(syn_score),
            "PPL": float(ppl),
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "saved_at": datetime.now().isoformat()
        }
        results.append(rec)

        # write per-episode JSON
        per_out = out_dir / f"aradawer_{re.sub(r'[^0-9a-zA-Z-_]', '_', title)}.json"
        per_out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save aggregated CSV/JSON
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(results)
        csv_out = out_dir / "aradawer_results.csv"
        df.to_csv(csv_out, index=False, encoding="utf-8")
    else:
        json_out = out_dir / "aradawer_results.json"
        json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compute correlations: SUB/DEL/INS with SEM_SIM and SYN_SCORE
    if results:
        arr = results
        subs = np.array([r['SUB'] for r in arr], dtype=float)
        dels = np.array([r['DEL'] for r in arr], dtype=float)
        ins = np.array([r['INS'] for r in arr], dtype=float)
        sems = np.array([r['SEM_SIM'] for r in arr], dtype=float)
        syns = np.array([r['SYN_SCORE'] for r in arr], dtype=float)

        def corr_print(x, y, labelx, labely):
            try:
                c, p = stats.pearsonr(x, y)
                print(f"Pearson({labelx},{labely}) = {c:.4f} (p={p:.3e})")
            except Exception as e:
                print(f"Pearson({labelx},{labely}) failed: {e}")

        print("\n--- Correlations between WER components and semantic/syntactic scores ---")
        for metric, arr_metric in [("SUB", subs), ("DEL", dels), ("INS", ins)]:
            corr_print(arr_metric, sems, metric, "SEM_SIM")
            corr_print(arr_metric, syns, metric, "SYN_SCORE")

        wer_vals = np.array([r['WER'] for r in arr], dtype=float)
        print(f"\nOverall mean WER: {wer_vals.mean():.4f}")

    # UMAP
    if all_embeddings:
        emb_matrix = np.vstack(all_embeddings)
        umap_out = out_dir / "umap_ref_vs_hyp.png"
        make_umap_plot(emb_matrix, all_labels, umap_out)
        logging.info("Saved UMAP plot to %s", umap_out)

    logging.info("Saved aggregated results to %s", out_dir)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AraDiaWER-style pipeline runner (predictions provided in --pred_dir)")
    p.add_argument("--json", required=True, help="Path to SCRAPE/main.json (list of episodes)")
    p.add_argument("--ref_dir", required=True, help="Directory with reference transcripts named <title>.txt")
    p.add_argument("--pred_dir", required=True, help="Directory with predicted transcripts (Files named <title>.txt)")
    p.add_argument("--out_dir", default="aradawer_out", help="Output directory")
    p.add_argument("--embed_model", default="sentence-transformers/LaBSE", help="SentenceTransformer model for semantic similarity (default: LaBSE)")
    p.add_argument("--lm_model", default=None, help="Causal LM for perplexity (e.g. aubmindlab/aragpt2-base). If omitted, PPL is skipped.")
    p.add_argument("--use_camel", action="store_true", help="Use CAMeL Tools if available for morphological syntactic scoring")
    args = p.parse_args()

    try:
        process_all(args)
    except Exception as e:
        logging.exception("Pipeline failed: %s", e)
        raise
