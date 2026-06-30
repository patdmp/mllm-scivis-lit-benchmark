# Benchmarking MLLMs for Scientific Visualization Literacy

This repository contains the data and model predictions for the paper:

> **Benchmarking Multimodal Large Language Models for Scientific Visualization Literacy**

---

## Repository Structure

```
.
├── SVLAT/
│   ├── svlat.csv                     # SVLAT benchmark: 49 items with metadata, questions, answers
│   └── svlat-visualizations/         # Visualization stimuli ( images and animations)
├── outputs/
│   ├── close_source_models/          # GPT-5.4, Claude-Opus-4.6, Gemini-3.1-Pro-Preview (CSV, 10 runs each)
│   ├── open_source_models/           # LLaVA-OneVision-1.5, InternVL3.5, Qwen3.5 (CSV, 10 runs each)
│   └── raw/                          # Raw per-item JSON responses (open-source models)
│       ├── llava/                    # Raw per-item JSON responses — LLaVA
│       ├── internvl/                 # Raw per-item JSON responses — InternVL
│       └── qwen/                     # Raw per-item JSON responses — Qwen
├── src/
│   ├── evaluate.py                   # Run zero-shot evaluation on open-source models
│   ├── utils.py                      # Shared utilities (prompt building, response parsing)
│   ├── convert_open_source_to_csv.py # Convert JSON results to CSV format
│   └── download_models.py            # Download open-source model weights
├── environment.yml                   # Conda environment specification
└── requirements.txt                  # pip dependencies
```

---

## Setup

### Option A — Conda

```bash
conda env create -f environment.yml
conda activate ai-scivis
```

### Option B — pip

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

### 1. Re-run model evaluation (open-source models)

> Requires a GPU and model weights (~8–9B parameters each).

```bash
# Download model weights first
python src/download_models.py

# Run all open-source models, 10 runs each
python src/evaluate.py

# Convert JSON output to CSV
python src/convert_open_source_to_csv.py
```

---

## Result File Format

### CSV files (`outputs/close_source_models/` and `outputs/open_source_models/`)

Each CSV row is one benchmark item, with columns for each of 10 independent runs:

| Column | Description |
|--------|-------------|
| `item_num` | Item index |
| `vis_file` | Visualization filename |
| `question` | Question text |
| `{model}_score_{k}` | 1 = correct, 0 = incorrect (run k) |
| `{model}_predicted_{k}` | Model's chosen answer letter (run k) |
| `{model}_rationale_{k}` | Model's reasoning text (run k) |
| `{model}_correct_{k}` | The correct answer letter for that run's shuffled options |

### JSON files (`outputs/raw/{llava,internvl,qwen}/item{XX}_run{YY}.json`)

Per-item, per-run response from open-source models. Includes shuffled option mapping, raw response, parsed choice, rationale, and correctness flag.

---

## Models Evaluated

| Model | Type | Parameter count |
|-------|------|----------------|
| GPT-5.4 (`gpt-5.4`) | Closed-source | — |
| Claude-Opus-4.6 (`claude-opus-4-6`) | Closed-source | — |
| Gemini-3.1-Pro-Preview (`gemini-3.1-pro-preview`) | Closed-source | — |
| LLaVA-OneVision-1.5 (`lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`) | Open-source | 8B |
| InternVL3.5 (`OpenGVLab/InternVL3_5-8B`) | Open-source | 8B |
| Qwen3.5 (`Qwen/Qwen3.5-9B`) | Open-source | 9B |

Human baseline: N = 485 participants via Prolific (Do et al., 2026).

---

## Notes on Closed-Source Model Evaluation

GPT-5.4 and Claude-Opus-4.6 do not support direct video input. For animation items (MP4), frames were extracted and provided as static images. Gemini-3.1-Pro-Preview natively supports video and received MP4 inputs directly.

All models were evaluated in zero-shot mode with a fixed JSON response format. Answer options were randomly shuffled across 10 independent runs (seeded) to control for positional bias.

---

## Citation

The SVLAT benchmark items and stimuli used in this study are from:

```bibtex
@article{do2026svlat,
  title   = {{SVLAT}: Scientific Visualization Literacy Assessment Test},
  author  = {Patrick Phuoc Do and Kaiyuan Tang and Kuangshi Ai and Chaoli Wang},
  journal = {arXiv preprint arXiv:2603.19000},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.19000} 
}
```

Supplementary materials for SVLAT are available at: https://osf.io/hr3nw/

---

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to share and adapt the materials for non-commercial purposes, provided you give appropriate credit and distribute any derivative work under the same license.
