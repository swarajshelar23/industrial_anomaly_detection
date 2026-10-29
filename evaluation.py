from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from anomaly_detection import build_heatmap, compute_image_score, compute_reconstruction_error, compute_threshold_from_validation
from config import DATASET_ROOT, IMAGE_SIZE, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, THRESHOLD_PERCENTILE, ensure_directories
from data_loader import build_category_dataset
from models import load_model_keras
from preprocessing import load_mask, load_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained anomaly detection model on one category or all categories.")
    parser.add_argument("--category", type=str, default=None, help="Category to evaluate.")
    parser.add_argument("--model", type=str, default="ae_cnn", choices=["ae_cnn", "denoising_ae_cnn", "vae_cnn"], help="Model name to evaluate.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT), help="Root directory of the MVTec AD dataset.")
    parser.add_argument("--all-categories", action="store_true", help="Evaluate all categories.")
    return parser.parse_args()


def image_scores_for_category(category: str, model_name: str, dataset_root: str, image_size: int = IMAGE_SIZE):
    dataset = build_category_dataset(category=category, dataset_root=dataset_root)
    model_path = MODELS_DIR / f"{category}_{model_name}.keras"
    model = load_model_keras(str(model_path))

    scores, labels = [], []
    for img_path in dataset.test_good:
        image = load_rgb_image(img_path, target_size=(image_size, image_size)).astype(np.float32)
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        error_map = compute_reconstruction_error(image, pred)
        scores.append(float(np.mean(error_map)))
        labels.append(0)

    for defect_name, paths in dataset.defect_images.items():
        for img_path in paths:
            image = load_rgb_image(img_path, target_size=(image_size, image_size)).astype(np.float32)
            pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            error_map = compute_reconstruction_error(image, pred)
            scores.append(float(np.mean(error_map)))
            labels.append(1)

    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def evaluate_one_category(category: str, model_name: str, dataset_root: str, image_size: int = IMAGE_SIZE, threshold_percentile: float = THRESHOLD_PERCENTILE):
    dataset = build_category_dataset(category=category, dataset_root=dataset_root)
    scores, labels = image_scores_for_category(category, model_name, dataset_root, image_size)
    if len(scores) < 2:
        raise ValueError(f"Not enough images to compute evaluation metrics for {category} / {model_name}")

    validation_scores = []
    for path in dataset.train_good[: min(20, len(dataset.train_good))]:
        image = load_rgb_image(path, target_size=(image_size, image_size)).astype(np.float32)
        model = load_model_keras(str(MODELS_DIR / f"{category}_{model_name}.keras"))
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        error_map = compute_reconstruction_error(image, pred)
        validation_scores.append(float(np.mean(error_map)))

    threshold = compute_threshold_from_validation(validation_scores, percentile=threshold_percentile)
    predictions = (scores > threshold).astype(int)

    auc = roc_auc_score(labels, scores)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    acc = accuracy_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metric_row = {
        "category": category,
        "model": model_name,
        "roc_auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "threshold": float(threshold),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    metrics_path = METRICS_DIR / f"{category}_{model_name}_metrics.csv"
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False)
    return metric_row


def main() -> None:
    args = parse_args()
    ensure_directories()
    categories = [args.category] if args.category else ["bottle", "cable", "capsule", "hazelnut"]
    if not args.all_categories and args.category is None:
        categories = ["bottle"]

    all_results = []
    for category in categories:
        result = evaluate_one_category(category, args.model, args.dataset_root, image_size=IMAGE_SIZE)
        all_results.append(result)

    summary_path = METRICS_DIR / "evaluation_summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print(pd.DataFrame(all_results).to_string(index=False))
    print(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
