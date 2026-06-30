"""
Download vision-language models from HuggingFace to the models/ directory.

Models downloaded:
  - LLaVA-OneVision-1.5-8B-Instruct  (lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)
  - InternVL3.5-8B                   (OpenGVLab/InternVL3_5-8B)
  - Qwen3.5-9B                       (Qwen/Qwen3.5-9B)

Requirements:
  pip install huggingface_hub
"""

import os
import shutil
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "llava": {
        "repo_id": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "description": "LLaVA-OneVision-1.5 8B Instruct",
    },
    "internvl": {
        "repo_id": "OpenGVLab/InternVL3_5-8B",
        "description": "InternVL3.5 8B",
    },
    "qwen3.5": {
        "repo_id": "Qwen/Qwen3.5-9B",
        "description": "Qwen3.5 9B",
    },
}

MODELS_DIR = Path(__file__).parent.parent / "models"


def remove_model(name: str) -> None:
    local_dir = MODELS_DIR / name
    if local_dir.exists():
        print(f"Removing {local_dir} ...")
        shutil.rmtree(local_dir)
        print(f"Removed {local_dir}")
    else:
        print(f"Nothing to remove: {local_dir} does not exist")


def download_model(name: str, repo_id: str, description: str, token: str | None = None) -> None:
    local_dir = MODELS_DIR / name
    print(f"\n{'='*60}")
    print(f"Downloading {description}")
    print(f"  Source : {repo_id}")
    print(f"  Dest   : {local_dir}")
    print(f"{'='*60}")

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )

    print(f"Done: {description} saved to {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VLMs from HuggingFace")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to download (default: all)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing model directories before downloading",
    )
    args = parser.parse_args()

    targets = list(MODELS.keys()) if "all" in args.models else args.models

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in targets:
        if args.clean:
            remove_model(name)
        cfg = MODELS[name]
        download_model(name, cfg["repo_id"], cfg["description"], token=args.token)

    print("\nAll requested models downloaded successfully.")


if __name__ == "__main__":
    main()
