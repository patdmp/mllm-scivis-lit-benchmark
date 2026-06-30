"""
Shared utilities: CSV loading, option shuffling, prompt building,
frame extraction, and JSON response parsing.
"""

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LETTERS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = """\
You are a scientific visualization expert. You will be given a scientific \
visualization (image or animation), its caption, and a question about it.

Provide your response in the following JSON format:
{
  "choice": "<the selected answer option (e.g., 'A', 'B', 'C', 'D') or 'Not sure'>",
  "answer_value": "<the full text of the selected answer option>",
  "rationale": "<your reason for choosing this answer, in 5 sentences or fewer>"
}

Requirements:
1. Base your answer strictly on the provided visualization and caption. Do not use prior knowledge or make assumptions.
2. Keep your rationale concise, with a maximum of 5 sentences.
3. If the answer cannot be determined confidently from the provided visualization and caption, set "choice" to "Not sure" and explain why in your rationale.
4. Return valid JSON only. Do not include Markdown or any extra text outside the JSON object.\
"""


# ──────────────────────────────────────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────────────────────────────────────

def load_items(csv_path: Path) -> list[dict]:
    """Return list of item dicts parsed from svlat.csv."""
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty trailing rows
            if not row.get("Item ID", "").strip():
                continue
            items.append({
                "item_id":    row["Item ID"].strip(),
                "vis":        row["Vis"].strip(),
                "technique":  row["Visualization Technique"].strip(),
                "vis_format": row["Vis Format"].strip(),   # "Image" or "Video"
                "n_choices":  int(row["# Answer Choice"].strip()),
                "answer_idx": int(row["Answer"].strip()),  # 1-indexed
                "task":       row["Task"].strip(),
                "vis_file":   row["vis_file"].strip(),
                "caption":    row["caption"].strip(),
                "options": [
                    row["option_1"].strip(),
                    row["option_2"].strip(),
                    row.get("option_3", "").strip(),
                    row.get("option_4", "").strip(),
                ],
                "question":   row["question"].strip(),
            })
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Option shuffling
# ──────────────────────────────────────────────────────────────────────────────

def shuffle_options(item: dict, seed: int) -> tuple[dict, str]:
    """
    Randomly permute the answer choices for one item/run.

    Returns:
        shuffled_options: {"A": {"original_idx": int (1-based), "text": str}, ...}
        correct_letter:   the letter (A/B/C/D) that corresponds to the correct answer
    """
    n = item["n_choices"]
    # Build list of (1-based original index, text) for the valid options
    original = [(i + 1, item["options"][i]) for i in range(n)]

    rng = random.Random(seed)
    shuffled = original[:]
    rng.shuffle(shuffled)

    letters = LETTERS[:n]
    shuffled_options: dict[str, dict] = {}
    correct_letter = ""
    for letter, (orig_idx, text) in zip(letters, shuffled):
        shuffled_options[letter] = {"original_idx": orig_idx, "text": text}
        if orig_idx == item["answer_idx"]:
            correct_letter = letter

    return shuffled_options, correct_letter


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────────────────────

def build_user_text(item: dict, shuffled_options: dict) -> str:
    """
    Build the text portion of the user message (caption + question + options).
    The system prompt is handled separately by each model runner.
    """
    options_lines = "\n".join(
        f"{letter}. {info['text']}"
        for letter, info in shuffled_options.items()
    )
    return (
        f"Caption: {item['caption']}\n\n"
        f"Question: {item['question']}\n\n"
        f"Answer options:\n{options_lines}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Frame extraction for video items
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: Path, n_frames: int) -> list[Image.Image]:
    """
    Extract n_frames evenly-spaced frames from a video as PIL Images (RGB).
    Falls back to a black placeholder if cv2 is unavailable or the file fails.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        raise ImportError("opencv-python is required for video items. Run: pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1  # fallback

    # Compute evenly-spaced frame indices
    if n_frames == 1:
        indices = [total // 2]
    else:
        indices = [int(total * i / (n_frames - 1)) for i in range(n_frames)]
        indices[-1] = min(indices[-1], total - 1)  # clamp last

    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Retry from current position
            ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            # Placeholder black frame
            frames.append(Image.new("RGB", (224, 224), color=0))

    cap.release()
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict[str, Any] | None:
    """
    Robustly extract the JSON object from a model response.

    Handles:
    - Clean JSON output
    - JSON wrapped in markdown code fences (```json ... ```)
    - Thinking blocks (<think>...</think>) before the JSON
    - Extra text before/after the JSON
    """
    if not raw:
        return None

    # Strip thinking blocks (Qwen3.5 may emit these even with enable_thinking=False)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Unescape markdown-escaped underscores (LLaVA outputs "answer\_value")
    text = text.replace("\\_", "_")

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Extract the first balanced {...} block (handles extra text after the closing brace)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def normalize_choice(parsed: dict | None, shuffled_options: dict) -> str | None:
    """
    Return the letter ('A','B','C','D', or 'Not sure') from a parsed response,
    normalizing common model quirks (e.g., lowercase, parentheses, full answer text).
    Returns None if it cannot be determined.
    """
    if parsed is None:
        return None

    choice = str(parsed.get("choice", "")).strip()

    # Direct match
    if choice in shuffled_options:
        return choice
    if choice.lower() == "not sure":
        return "Not sure"

    # Uppercase single letter
    upper = choice.upper().strip("(). ")
    if upper in shuffled_options:
        return upper

    # Model may have responded with the full answer text — match against options
    answer_value = str(parsed.get("answer_value", "")).strip().lower()
    for letter, info in shuffled_options.items():
        if answer_value and info["text"].strip().lower() == answer_value:
            return letter

    return None
