"""
Main evaluation script.

For each model × item × run (10 runs):
  1. Shuffle answer choices with seed = run_index (0–9)
  2. Run inference
  3. Save result to outputs/raw/{model}/{item_id}_run{run:02d}.json

Usage:
    python src/evaluate.py                          # all models, 10 runs
    python src/evaluate.py --models llava internvl  # specific models
    python src/evaluate.py --models qwen --runs 3   # 3 runs only
    python src/evaluate.py --images-only            # image items only
    python src/evaluate.py --resume                 # skip already-completed runs
    python src/evaluate.py --quantize 4bit          # 4-bit quantization (large models)
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure Unicode output works on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

# ── project paths ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "outputs"
VIS_DIR    = ROOT / "SVLAT" / "svlat-visualizations"
CSV_PATH   = ROOT / "SVLAT" / "svlat.csv"
MODELS_DIR = ROOT / "models"

# Add src/ to path so imports work when run from repo root
sys.path.insert(0, str(ROOT / "src"))

from utils import (
    load_items,
    shuffle_options,
    build_user_text,
    extract_frames,
    parse_json_response,
    normalize_choice,
)
from models import RUNNERS


# ──────────────────────────────────────────────────────────────────────────────
# Result I/O
# ──────────────────────────────────────────────────────────────────────────────

def result_path(model_name: str, item_id: str, run: int) -> Path:
    return RESULTS_DIR / "raw" / model_name / f"{item_id}_run{run:02d}.json"


def save_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_result(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Load visualization
# ──────────────────────────────────────────────────────────────────────────────

def load_visual(item: dict, n_frames: int) -> tuple[list[Image.Image], bool]:
    """
    Returns (images, is_video).
    For image items: ([PIL image], False).
    For video items: ([frame1, frame2, ...], True).
    """
    vis_path = VIS_DIR / item["vis_file"]
    is_video = item["vis_format"].lower() == "video"

    if is_video:
        frames = extract_frames(vis_path, n_frames=n_frames)
        return frames, True
    else:
        img = Image.open(vis_path).convert("RGB")
        return [img], False


# ──────────────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────────────

def run_single(runner, item: dict, run: int) -> dict:
    """Run one item × one run. Returns the full result dict."""
    seed = run  # seeds 0–9 give different option orderings

    shuffled_options, correct_letter = shuffle_options(item, seed)
    user_text = build_user_text(item, shuffled_options)

    images, is_video = load_visual(item, n_frames=runner.N_VIDEO_FRAMES)

    try:
        raw_response = runner.run(images, user_text, is_video)
        parse_error = None
    except Exception as e:
        raw_response = ""
        parse_error = traceback.format_exc()
        print(f"    [ERROR] inference failed: {e}")

    parsed = parse_json_response(raw_response)
    chosen_letter = normalize_choice(parsed, shuffled_options)
    is_correct = (chosen_letter == correct_letter) if chosen_letter not in (None, "Not sure") else False

    return {
        "item_id":               item["item_id"],
        "model":                 runner.__class__.__name__.replace("Runner", "").lower(),
        "run":                   run,
        "seed":                  seed,
        "n_choices":             item["n_choices"],
        "shuffled_options":      shuffled_options,
        "correct_answer_original_idx": item["answer_idx"],
        "correct_letter":        correct_letter,
        "raw_response":          raw_response,
        "parsed":                parsed,
        "chosen_letter":         chosen_letter,
        "is_correct":            is_correct,
        "parse_error":           parse_error,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    items: list[dict],
    n_runs: int,
    resume: bool,
) -> None:
    # Map model key → actual directory name under models/
    MODEL_DIR_NAMES = {"llava": "llava", "internvl": "internvl", "qwen": "qwen3.5"}
    model_dir = MODELS_DIR / MODEL_DIR_NAMES.get(model_name, model_name)
    RunnerClass = RUNNERS[model_name]
    runner = RunnerClass(model_dir)
    runner.load()

    total_items = len(items)
    for item_idx, item in enumerate(items):
        item_id = item["item_id"]

        if runner.IMAGE_ONLY and item["vis_format"].lower() == "video":
            print(f"\n  [{model_name}] Item {item_idx+1}/{total_items}: {item_id} — skipped (video, image-only model)")
            continue

        print(f"\n  [{model_name}] Item {item_idx+1}/{total_items}: {item_id}")

        for run in range(n_runs):
            rpath = result_path(model_name, item_id, run)

            if resume and rpath.exists():
                existing = load_result(rpath)
                if existing is not None and existing.get("raw_response", ""):
                    status = "correct" if existing.get("is_correct") else "wrong"
                    print(f"    run {run:02d} → skipped (already done, {status})")
                    continue

            result = run_single(runner, item, run)
            save_result(rpath, result)

            chosen = result["chosen_letter"] or "?"
            correct = result["correct_letter"]
            ok = "✓" if result["is_correct"] else "✗"
            opts = result["shuffled_options"]
            chosen_text  = opts[chosen]["text"]  if chosen in opts else "—"
            correct_text = opts[correct]["text"] if correct in opts else "—"
            print(f"    run {run:02d} {ok}  chosen={chosen}: {chosen_text!r}  |  correct={correct}: {correct_text!r}")

    runner.unload()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate open-source VLMs on SVLAT")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(RUNNERS.keys()) + ["all"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per item (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose result JSON already exists",
    )
    parser.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="Evaluate only these item IDs (e.g. item01 item02)",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Evaluate only image items",
    )
    args = parser.parse_args()

    items = load_items(CSV_PATH)
    if args.images_only:
        items = [it for it in items if it["vis_format"].lower() == "image"]
        print(f"Filtered to {len(items)} image items.")
    if args.items:
        items = [it for it in items if it["item_id"] in args.items]
        print(f"Filtered to {len(items)} items: {[it['item_id'] for it in items]}")

    model_names = list(RUNNERS.keys()) if "all" in args.models else args.models

    print(f"Models : {model_names}")
    print(f"Items  : {len(items)}")
    print(f"Mode   : {'image-only' if args.images_only else 'all items'}")
    print(f"Runs   : {args.runs}")
    print(f"Resume : {args.resume}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f" Model: {model_name}")
        print(f"{'='*60}")
        evaluate_model(model_name, items, args.runs, args.resume)

    print("\nDone. Run `python src/fill_xlsx.py` to populate the results spreadsheet.")


if __name__ == "__main__":
    main()
