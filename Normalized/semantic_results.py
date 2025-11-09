"""
semantic_results.py

Place this file inside the `Normalized/` directory and run it from there.

Usage:
  - (Optional) Set environment variable OPENROUTER_API_KEY to your OpenRouter API key if you want LLM checks.
  - Run: python semantic_results.py

This version assumes it's run from the `Normalized/` folder and will:
  - load `mainn.json` from the same folder
  - scan other `.json` files in the same folder (excluding `mainn.json` and `semantic_results.json`)
  - compute word-level alignments and WER, and optionally call OpenRouter model
  - write `semantic_results.json` into the same folder

Requirements: Python 3.8+, requests
"""

import os
import json
import glob
import time
import argparse
import difflib
from typing import List, Dict, Any, Optional

import requests

# When placed in Normalized/, run from that folder. Root is script directory.
ROOT = os.path.abspath(os.path.dirname(__file__))
GROUND_TRUTH = os.path.join(ROOT, "mainn.json")
OUTPUT_FILE = os.path.join(ROOT, "semantic_results.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_json_files(root: str) -> List[str]:
    # Only include the hypothesis files we care about (case-sensitive): GPT4.json and Whisper.json
    candidates = [os.path.join(root, "GPT4.json"), os.path.join(root, "Whisper.json")]
    existing = [os.path.abspath(p) for p in candidates if os.path.exists(p)]
    return existing


def guess_transcript_key(item: dict) -> Optional[str]:
    for k in ( "transcribed_ar", "transcript", "text", "transcription"):
        if k in item:
            return k
    for k, v in item.items():
        if isinstance(v, str) and len(v.split()) > 3:
            return k
    return None


def words(s: str) -> List[str]:
    if s is None:
        return []
    return [w for w in s.strip().split() if w != ""]


def align_words(ref: List[str], hyp: List[str]) -> List[Dict[str, Any]]:
    sm = difflib.SequenceMatcher(a=ref, b=hyp)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        out.append({
            "op": tag,
            "ref_tokens": ref[i1:i2],
            "hyp_tokens": hyp[j1:j2],
            "ref_pos": (i1, i2),
            "hyp_pos": (j1, j2),
        })
    return out


def wer_from_alignment(alignment: List[Dict[str, Any]], ref_len: int) -> float:
    S = sum(len(a["ref_tokens"]) for a in alignment if a["op"] == "replace")
    D = sum(len(a["ref_tokens"]) for a in alignment if a["op"] == "delete")
    I = sum(len(a["hyp_tokens"]) for a in alignment if a["op"] == "insert")
    if ref_len == 0:
        return float('nan')
    return (S + D + I) / ref_len


def levenshtein(a: str, b: str) -> int:
    # simple iterative DP Levenshtein distance (works on characters)
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            ins = cur[j-1] + 1
            delete = prev[j] + 1
            subs = prev[j-1] + (0 if ca == cb else 1)
            cur[j] = min(ins, delete, subs)
        prev = cur
    return prev[lb]


def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    ref_chars = list(ref)
    if len(ref_chars) == 0:
        return float('nan')
    dist = levenshtein(ref, hyp)
    return dist / len(ref_chars)


def call_openrouter_check(api_key: str, model: str, reference: str, candidate: str, timeout=20.0) -> Dict[str, Any]:
    # Use the same API base path the OpenRouter SDK/client uses (works in Normalized/test.py)
    url = "https://openrouter.ai/api/v1/chat/completions"
    # include Accept to prefer JSON responses and mirror SDK behavior
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    system = (
        "You are a strict normalizer/orthography judge for Arabic (Lebanese dialect)."
        "\nTask: given a single reference token and a single candidate token, output EXACTLY one JSON object"
        " with no explanations, reasoning, or extra fields. Do NOT produce chain-of-thought or any commentary."
        "\nAllowed verdict values: \"same\", \"variant\", \"different\", \"error\"."
        "\nConfidence must be a number between 0.0 and 1.0."
  
        "Examples: {\"verdict\":\"same\",\"confidence\":0.98}"
        "{\"verdict\":\"different\",\"confidence\":0.98}"
    )
    user = (
        f"REFERENCE: {reference}\nCANDIDATE: {candidate}\n"
        "Respond with ONLY the exact JSON object as specified in the system message."
    )
    # Only try the exact requested model (no fallback). If it fails or returns non-JSON, return error.
    try_models = [model]

    # prepare the static part of the payload and set model per attempt
    # increase token budget and make generation deterministic to avoid truncation and extra reasoning
    base_payload = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 900,
    }
    last_raw = ""
    for model_id in try_models:
        payload = dict(base_payload)
        payload["model"] = model_id
        try:
            print(f"[llm][http] requesting model={model_id} to {url}")
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            # capture raw response text for diagnostics
            raw = r.text if hasattr(r, 'text') else ''
            last_raw = raw
            # log status and content-type early for clearer diagnostics
            try:
                status = r.status_code
                ctype = r.headers.get("Content-Type", "")
                print(f"[llm][http] status={status} content-type={ctype}")
                if "application/json" not in ctype.lower():
                    print(f"[llm][raw response] {raw[:1000]}")
                    # do NOT try other models; return error to caller so the run fails fast and transparently
                    return {"verdict": "error", "confidence": 0.0, "reason": "non_json_response", "raw": raw, "model_used": model_id}
            except Exception:
                pass
            try:
                r.raise_for_status()
            except Exception as e:
                # include raw body to help debugging
                return {"verdict": "error", "confidence": 0.0, "reason": f"http_error: {e}", "raw": raw}
            # try to parse json payload first
            try:
                data = r.json()
            except Exception:
                # not JSON — return error immediately
                print(f"[llm][raw response] {raw[:1000]}")
                return {"verdict": "error", "confidence": 0.0, "reason": "invalid_json", "raw": raw, "model_used": model_id}
            content = None
            if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                # choice may contain message (with content) or direct text
                msg = choice.get("message", {}) if isinstance(choice, dict) else {}
                content = (msg.get("content") if isinstance(msg, dict) else None) or choice.get("text")
                # fallback: some providers include chain-of-thought or reasoning fields
                if not content:
                    # try 'reasoning' field
                    reasoning = (msg.get("reasoning") if isinstance(msg, dict) else None) or choice.get("reasoning")
                    if reasoning:
                        content = reasoning
                        print(f"[llm][raw reasoning] {str(reasoning)[:1000]}")
                    else:
                        rd = (msg.get("reasoning_details") if isinstance(msg, dict) else None) or choice.get("reasoning_details")
                        if isinstance(rd, list) and rd:
                            # join text fragments if present
                            joined = " ".join((d.get("text") if isinstance(d, dict) else str(d)) for d in rd)
                            content = joined
                            print(f"[llm][raw reasoning_details] {joined[:1000]}")
            if not content:
                print(f"[llm][raw response] {raw[:1000]}")
                return {"verdict": "error", "confidence": 0.0, "reason": "empty_content", "raw": raw, "model_used": model_id}
            text = str(content).strip()
            # log the model's content for debugging
            print(f"[llm][raw content] {text}")
            # try to find JSON object in the content
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                jtext = text[start:end+1]
                try:
                    parsed = json.loads(jtext)
                    # attach raw content for transparency
                    parsed.setdefault('raw', text)
                    parsed.setdefault('model_used', model_id)
                    return parsed
                except Exception:
                    # try next model
                    continue
            # If we couldn't parse JSON, try next model
            continue
        except Exception as e:
            # store last raw and continue or return if this was a critical failure
            last_raw = str(e)
            continue

    # All models tried and failed
    return {"verdict": "error", "confidence": 0.0, "reason": "all_models_failed_or_non_json", "raw": last_raw}


def process_pair(ref_text: str, hyp_text: str, llm_enabled: bool, api_key: Optional[str], model: str, max_llm_calls: int, llm_delay: float) -> Dict[str, Any]:
    ref_words = words(ref_text)
    hyp_words = words(hyp_text)
    alignment = align_words(ref_words, hyp_words)
    wer = wer_from_alignment(alignment, len(ref_words))
    errors = []
    llm_calls = 0
    for a in alignment:
        if a["op"] == "equal":
            continue
        entry = {"op": a["op"], "ref_tokens": a["ref_tokens"], "hyp_tokens": a["hyp_tokens"], "ref_pos": a["ref_pos"], "hyp_pos": a["hyp_pos"], "llm": None}
        if llm_enabled and api_key and llm_calls < max_llm_calls:
            pairs = []
            if a["op"] == "replace":
                n = max(len(a["ref_tokens"]), len(a["hyp_tokens"]))
                for i in range(n):
                    ref_t = a["ref_tokens"][i] if i < len(a["ref_tokens"]) else ""
                    hyp_t = a["hyp_tokens"][i] if i < len(a["hyp_tokens"]) else ""
                    pairs.append((ref_t, hyp_t))
            elif a["op"] == "insert":
                for hyp_t in a["hyp_tokens"]:
                    pairs.append(("", hyp_t))
            elif a["op"] == "delete":
                for ref_t in a["ref_tokens"]:
                    pairs.append((ref_t, ""))
            llm_results = []
            for ref_t, hyp_t in pairs:
                if llm_calls >= max_llm_calls:
                    break
                if ref_t == "" and hyp_t == "":
                    continue
                resp = call_openrouter_check(api_key, model, ref_t, hyp_t)
                llm_results.append({"ref": ref_t, "hyp": hyp_t, "resp": resp})
                llm_calls += 1
                time.sleep(llm_delay)
            entry["llm"] = llm_results
        errors.append(entry)
    return {"wer": wer, "ref_len": len(ref_words), "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default=GROUND_TRUTH, help="path to mainn.json ground truth")
    parser.add_argument("--out", default=OUTPUT_FILE, help="output json file")
    parser.add_argument("--model", default="openai/gpt-oss-20b:free", help="OpenRouter model id")
    parser.add_argument("--max-llm-calls", type=int, default=500, help="max total LLM calls (per run)")
    parser.add_argument("--llm-delay", type=float, default=0.35, help="seconds to wait between LLM calls")
    parser.add_argument("--dry-run-llm", action="store_true", help="do not call LLM even if key is present")
    args = parser.parse_args()

    gt_path = os.path.abspath(args.ground_truth)
    if not os.path.exists(gt_path):
        print(f"Ground truth not found at {gt_path}")
        return
    ground = load_json(gt_path)
    gt_map = {}
    for item in ground:
        title = item.get("title")
        if title:
            gt_map[title.strip()] = item

    json_files = find_json_files(ROOT)
    # support multiple possible env var names for API key (user may store it as OPENROUTER_API_KEY or LLM_API_KEY)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    llm_enabled = bool(api_key) and not args.dry_run_llm
    if not api_key:
        print("OPENROUTER_API_KEY not set; LLM checks will be skipped.")
    if args.dry_run_llm:
        print("--dry-run-llm set; LLM checks are disabled.")

    total_llm_calls = 0

    # Aggregation holders for WER/CER statistics
    overall_results = []
    per_file_acc = {}

    for path in sorted(json_files):
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skipping {path}: failed to load JSON: {e}")
            continue
        print(f"[file] Processing {os.path.relpath(path, ROOT)}")
        items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        file_res = {"path": path, "items": []}
        file_basename = os.path.basename(path)
        per_file_acc.setdefault(file_basename, {"sum_wer": 0.0, "sum_cer": 0.0, "count": 0, "results": []})
        for item in items:
            title = item.get("title")
            if not title:
                continue
            title_key = title.strip()
            gt_item = gt_map.get(title_key)
            if not gt_item:
                continue
            # For ground-truth we prefer the Arabic transcript field `transcribed_ar` explicitly.
            # This ensures WER is computed against the Arabic reference when available.
            gt_key = "transcribed_ar" if "transcribed_ar" in gt_item else guess_transcript_key(gt_item)
            hyp_key = guess_transcript_key(item)
            if not gt_key or not hyp_key:
                continue
            ref_text = gt_item.get(gt_key, "")
            hyp_text = item.get(hyp_key, "")
            remaining_calls = max(0, args.max_llm_calls - total_llm_calls)
            print(f"  [item] {title_key} -> comparing gt_key='{gt_key}' hyp_key='{hyp_key}'")
            # run alignment first
            ref_words = words(ref_text)
            hyp_words = words(hyp_text)
            alignment = align_words(ref_words, hyp_words)
            wer_val = wer_from_alignment(alignment, len(ref_words))
            cer_val = cer(ref_text, hyp_text)
            pr = {"wer": wer_val, "cer": cer_val, "ref_len": len(ref_words), "errors": []}

            # process alignment entries with incremental LLM calls and progress prints
            llm_calls_used = 0
            # For each replacement (wrong predicted word), call the LLM per paired token to judge semantic equivalence.
            for a in alignment:
                if a["op"] == "equal":
                    continue
                entry = {"op": a["op"], "ref_tokens": a["ref_tokens"], "hyp_tokens": a["hyp_tokens"], "ref_pos": a["ref_pos"], "hyp_pos": a["hyp_pos"], "llm": None}
                # Only perform LLM checks for replacements (prediction word aligned to a ref word but different).
                if a["op"] == "replace" and llm_enabled and api_key and total_llm_calls + llm_calls_used < args.max_llm_calls:
                    llm_results = []
                    n = max(len(a["ref_tokens"]), len(a["hyp_tokens"]))
                    for i in range(n):
                        if total_llm_calls + llm_calls_used >= args.max_llm_calls:
                            print("    [llm] Reached max LLM calls limit; skipping further LLM checks")
                            break
                        ref_t = a["ref_tokens"][i] if i < len(a["ref_tokens"]) else ""
                        hyp_t = a["hyp_tokens"][i] if i < len(a["hyp_tokens"]) else ""
                        # Only call LLM for actual predicted words (hyp token exists and differs from ref)
                        if hyp_t == "" or ref_t == hyp_t:
                            continue
                        print(f"    [llm] Checking ref='{ref_t}' hyp='{hyp_t}'")
                        resp = call_openrouter_check(api_key, args.model, ref_t, hyp_t)
                        print(f"    [llm] -> verdict={resp.get('verdict')} confidence={resp.get('confidence')}")
                        llm_results.append({"ref": ref_t, "hyp": hyp_t, "resp": resp})
                        llm_calls_used += 1
                        time.sleep(args.llm_delay)
                    entry["llm"] = llm_results
                # For insert/delete ops we do not call the LLM here (no paired predicted token for a ref in delete,
                # and inserts don't map to a ref token). They remain recorded as errors without LLM judgement.
                pr["errors"].append(entry)
            total_llm_calls += llm_calls_used
            file_res["items"].append({"title": title_key, "gt_key": gt_key, "hyp_key": hyp_key, "result": pr, "truth": ref_text, "pred": hyp_text})
            # compute semantic-correct counts: equal tokens + LLM 'same'/'variant' judgments for ref tokens
            total_ref_tokens = len(ref_words)
            equal_count = sum(len(a["ref_tokens"]) for a in alignment if a["op"] == "equal")
            llm_semantic = 0
            for err in pr.get("errors", []):
                llm_list = err.get("llm")
                if not llm_list:
                    continue
                for rec in llm_list:
                    ref_t = rec.get("ref", "")
                    resp = rec.get("resp") or {}
                    verdict = ""
                    if isinstance(resp, dict):
                        verdict = str(resp.get("verdict", "")).lower()
                    if ref_t and verdict in ("same", "variant"):
                        llm_semantic += 1

            semantic_correct = equal_count + llm_semantic
            semantic_rate = (semantic_correct / total_ref_tokens) if total_ref_tokens > 0 else 0.0

            # accumulate stats (WER/CER and semantic)
            per_file_acc[file_basename]["sum_wer"] += (pr["wer"] if not (pr["wer"] != pr["wer"]) else 0.0)
            per_file_acc[file_basename]["sum_cer"] += (pr["cer"] if not (pr["cer"] != pr["cer"]) else 0.0)
            per_file_acc[file_basename]["sum_semantic"] = per_file_acc[file_basename].get("sum_semantic", 0.0) + semantic_correct
            per_file_acc[file_basename]["sum_ref_tokens"] = per_file_acc[file_basename].get("sum_ref_tokens", 0) + total_ref_tokens
            per_file_acc[file_basename]["count"] += 1
            per_file_acc[file_basename]["results"].append({"title": title_key, "WER": pr["wer"], "CER": pr["cer"], "semantic_correct": semantic_correct, "semantic_rate": semantic_rate, "truth": ref_text, "pred": hyp_text})
            overall_results.append({"title": title_key, "file": file_basename, "WER": pr["wer"], "CER": pr["cer"], "semantic_correct": semantic_correct, "semantic_rate": semantic_rate, "truth": ref_text, "pred": hyp_text})
            # Stream this episode result immediately to an NDJSON file (one JSON object per line).
            try:
                ndjson_path = args.out + ".ndjson"
                with open(ndjson_path, "a", encoding="utf-8") as nf:
                    json.dump(overall_results[-1], nf, ensure_ascii=False)
                    nf.write("\n")
            except Exception as e:
                print(f"Warning: failed to append NDJSON for {title_key}: {e}")
        print(f"[file] Processed {file_basename} (items={len(file_res['items'])})")

    # compute per-file and overall aggregates
    total_files = len(per_file_acc)
    total_matched = sum(v["count"] for v in per_file_acc.values())
    sum_wer = sum(v["sum_wer"] for v in per_file_acc.values())
    sum_cer = sum(v["sum_cer"] for v in per_file_acc.values())
    average_WER = (sum_wer / total_matched) if total_matched > 0 else 0.0
    average_CER = (sum_cer / total_matched) if total_matched > 0 else 0.0

    per_file_stats = {}
    for fname, v in per_file_acc.items():
        cnt = v["count"]
        per_file_stats[fname] = {
            "avg_WER": round((v["sum_wer"] / cnt) if cnt > 0 else 0.0, 4),
            "avg_CER": round((v["sum_cer"] / cnt) if cnt > 0 else 0.0, 4),
            "count": cnt,
        }

    out_summary = {
        "total_files": total_files,
        "total_matched_episodes": total_matched,
        "average_WER": round(average_WER, 4),
        "average_CER": round(average_CER, 4),
        "per_file_stats": per_file_stats,
        "results": overall_results,
    }

    # write final WER-style results JSON (not streamed) to output
    with open(args.out, "w", encoding="utf-8") as of:
        json.dump(out_summary, of, ensure_ascii=False, indent=2)

    print(f"Wrote WER results summary to {args.out}")
    if llm_enabled:
        print(f"Total LLM calls made (approx): {total_llm_calls}")


if __name__ == "__main__":
    main()
