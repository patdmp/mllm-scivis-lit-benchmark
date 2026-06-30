"""
convert_open_source_to_csv.py

Reads per-item-per-run JSON results from outputs/raw/{model}/ for LLaVA, InternVL,
and Qwen and writes one CSV per model to outputs/open_source_models/, matching the column format
of the closed-source CSVs (results_claude.csv, results_gemini.csv, etc.).

Output columns:
    item_num, vis_file, question,
    {prefix}_correct_{1..10},
    {prefix}_predicted_{1..10},
    {prefix}_score_{1..10},
    {prefix}_rationale_{1..10}
"""

import csv
import json
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CSV_PATH    = ROOT / "SVLAT" / "svlat.csv"
RESULTS_DIR = ROOT / "outputs"
OUT_DIR     = RESULTS_DIR / "open_source_models"

MODELS = {
    "llava":    "llava",
    "internvl": "internvl",
    "qwen":     "qwen",
}

N_RUNS = 10


# ── helpers ────────────────────────────────────────────────────────────────────
def load_items(csv_path: Path) -> list[dict]:
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("Item ID", "").strip():
                continue
            items.append({
                "item_id":  row["Item ID"].strip(),
                "vis_file": row["vis_file"].strip(),
                "question": row["question"].strip(),
            })
    return items


def load_run(model_dir: Path, item_id: str, run: int) -> dict | None:
    """Load one item/run JSON; return None if missing."""
    # item_id is like "item01"; run is 0-based
    num = item_id.replace("item", "")          # "01"
    path = model_dir / f"item{num}_run{run:02d}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_row(item: dict, item_num: int, prefix: str,
              model_dir: Path) -> dict:
    row = {
        "item_num": item_num,
        "vis_file": item["vis_file"],
        "question": item["question"],
    }
    for run in range(N_RUNS):
        n = run + 1          # 1-based column index
        data = load_run(model_dir, item["item_id"], run)
        if data is None:
            correct   = ""
            predicted = ""
            score     = ""
            rationale = ""
        else:
            correct   = data.get("correct_letter", "")
            predicted = data.get("chosen_letter", "")
            is_correct = data.get("is_correct")
            score     = 1 if is_correct else 0
            parsed    = data.get("parsed") or {}
            rationale = parsed.get("rationale", "")

        row[f"{prefix}_correct_{n}"]   = correct
        row[f"{prefix}_predicted_{n}"] = predicted
        row[f"{prefix}_score_{n}"]     = score
        row[f"{prefix}_rationale_{n}"] = rationale
    return row


def write_csv(model_name: str, prefix: str, items: list[dict]) -> None:
    model_dir = RESULTS_DIR / "raw" / model_name
    if not model_dir.exists():
        print(f"[skip] {model_dir} not found")
        return

    out_path = OUT_DIR / f"results_{model_name}.csv"
    fieldnames = ["item_num", "vis_file", "question"]
    for col in ("correct", "predicted", "score", "rationale"):
        for n in range(1, N_RUNS + 1):
            fieldnames.append(f"{prefix}_{col}_{n}")

    rows = []
    for i, item in enumerate(items, start=1):
        rows.append(build_row(item, i, prefix, model_dir))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] {out_path}  ({len(rows)} items)")


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items(CSV_PATH)
    print(f"Loaded {len(items)} items from {CSV_PATH.name}")
    for model_name, prefix in MODELS.items():
        write_csv(model_name, prefix, items)


if __name__ == "__main__":
    main()
