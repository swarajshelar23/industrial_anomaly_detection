from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import DATASET_ROOT, METRICS_DIR, MODEL_COMPARISON_DIR, OUTPUTS_DIR, ensure_directories
from evaluation import evaluate_one_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare autoencoder models across all four categories.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT), help="Root directory of the MVTec AD dataset.")
    parser.add_argument("--all-categories", action="store_true", default=True)
    return parser.parse_args()


def compare_all_categories(dataset_root: str):
    ensure_directories()
    categories = ["bottle", "cable", "capsule", "hazelnut"]
    models = ["ae_cnn", "denoising_ae_cnn", "vae_cnn"]
    rows = []
    for category in categories:
        for model in models:
            try:
                result = evaluate_one_category(category, model, dataset_root)
                rows.append(result)
            except Exception as exc:
                print(f"Skipping {category} {model}: {exc}")
    summary_path = METRICS_DIR / "model_comparison_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved comparison table to {summary_path}")
    return df


if __name__ == "__main__":
    args = parse_args()
    compare_all_categories(args.dataset_root)
