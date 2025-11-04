import os
import json
from jiwer import wer, cer

# === CONFIG ===
TRUTH_FILE = r"mainn.json"
PREDICTIONS_FOLDER = r"."
OUTPUT_FILE = r"WER_Results.json"


# === Load Ground Truth ===
def load_truths(truth_file):
    """Load ground truth transcripts as {title: text}."""
    with open(truth_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    truth_map = {}
    for entry in data:
        title = entry.get("title", "").strip()
        text = entry.get("transcribed_ar", "").strip()
        if title and text:
            truth_map[title] = text
    return truth_map


# === Evaluate Folder ===
def evaluate_folder(pred_folder, truth_map):
    all_results = []
    per_file_stats = {}
    all_wers, all_cers = [], []

    files = [f for f in os.listdir(pred_folder) if f.endswith(".json")]
    for filename in files:
        path = os.path.join(pred_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)

        print(f"📂 Evaluating {filename} ...")

        file_wers, file_cers = [], []
        for entry in preds:
            title = entry.get("title", "").strip()
            pred_text = entry.get("transcribed_ar", "").strip()
            truth_text = truth_map.get(title, "")

            if not truth_text or not pred_text:
                continue  # skip if missing

            # --- Compute metrics ---
            w = wer(truth_text, pred_text)
            c = cer(truth_text, pred_text)

            all_wers.append(w)
            all_cers.append(c)
            file_wers.append(w)
            file_cers.append(c)

            all_results.append({
                "title": title,
                "file": filename,
                "WER": round(w, 4),
                "CER": round(c, 4),
                "truth": truth_text,
                "pred": pred_text
            })

        # Per-file averages
        if file_wers:
            per_file_stats[filename] = {
                "avg_WER": round(sum(file_wers) / len(file_wers), 4),
                "avg_CER": round(sum(file_cers) / len(file_cers), 4),
                "count": len(file_wers)
            }

    # Global averages
    overall_wer = sum(all_wers) / len(all_wers) if all_wers else 0
    overall_cer = sum(all_cers) / len(all_cers) if all_cers else 0

    summary = {
        "total_files": len(files),
        "total_matched_episodes": len(all_results),
        "average_WER": round(overall_wer, 4),
        "average_CER": round(overall_cer, 4),
        "per_file_stats": per_file_stats,
        "results": all_results
    }

    return summary


# === Save Results ===
def save_results(summary, output_file):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ WER results saved to {output_file}")


if __name__ == "__main__":
    truth_map = load_truths(TRUTH_FILE)
    summary = evaluate_folder(PREDICTIONS_FOLDER, truth_map)
    save_results(summary, OUTPUT_FILE)
