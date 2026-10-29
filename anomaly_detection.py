from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from config import ANOMALY_HEATMAPS_DIR, IMAGE_SIZE, MODELS_DIR, PREDICTIONS_DIR, THRESHOLD_PERCENTILE
from data_loader import build_category_dataset
from models import load_model_keras
from preprocessing import build_inference_dataset, load_rgb_image, predict_with_model


def compute_reconstruction_error(input_image: np.ndarray, reconstructed_image: np.ndarray) -> np.ndarray:
    return np.square(input_image - reconstructed_image).astype(np.float32)


def compute_image_score(error_map: np.ndarray) -> float:
    return float(np.mean(error_map))


def compute_threshold_from_validation(validation_scores: Sequence[float], percentile: float = THRESHOLD_PERCENTILE) -> float:
    if len(validation_scores) == 0:
        raise ValueError("Validation scores are empty; cannot compute threshold.")
    return float(np.percentile(np.asarray(validation_scores, dtype=np.float32), percentile))


def build_heatmap(error_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    heatmap = np.mean(error_map, axis=-1)
    if normalize:
        heatmap = heatmap - np.min(heatmap)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
    return heatmap.astype(np.float32)


def detect_anomalies_for_category(category: str, model_name: str, dataset_root: str, image_size: int = IMAGE_SIZE, threshold_percentile: float = THRESHOLD_PERCENTILE):
    dataset = build_category_dataset(category=category, dataset_root=dataset_root)
    model_path = MODELS_DIR / f"{category}_{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model for category {category} and model {model_name} not found: {model_path}")

    model = load_model_keras(str(model_path))
    train_paths = [str(p) for p in dataset.train_good]
    normal_val_scores = []
    normal_validation_paths = train_paths[: max(1, min(len(train_paths), 20))]

    for path in normal_validation_paths:
        image = load_rgb_image(path, target_size=(image_size, image_size)).astype(np.float32)
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        error = compute_reconstruction_error(image, pred)
        normal_val_scores.append(float(np.mean(error)))

    threshold = compute_threshold_from_validation(normal_val_scores, percentile=threshold_percentile)

    predictions = []
    for defect_name, paths in dataset.defect_images.items():
        for p in paths:
            image = load_rgb_image(p, target_size=(image_size, image_size)).astype(np.float32)
            pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            error_map = compute_reconstruction_error(image, pred)
            score = compute_image_score(error_map)
            anomaly = score > threshold
            heatmap = build_heatmap(error_map)
            predictions.append({
                "category": category,
                "model": model_name,
                "defect_name": defect_name,
                "image_path": str(p),
                "score": float(score),
                "threshold": float(threshold),
                "predicted_anomalous": bool(anomaly),
                "heatmap_path": str(ANOMALY_HEATMAPS_DIR / f"{category}_{model_name}_{p.stem}_heatmap.png") if hasattr(p, 'stem') else None,
            })

    predictions_path = PREDICTIONS_DIR / f"{category}_{model_name}_predictions.json"
    with predictions_path.open("w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)

    return {"threshold": threshold, "predictions": predictions, "model_path": str(model_path)}


if __name__ == "__main__":
    for category in ["bottle", "cable", "capsule", "hazelnut"]:
        try:
            result = detect_anomalies_for_category(category, "ae_cnn", str(Path(__file__).resolve().parent))
            print(category, result["threshold"])
        except Exception as exc:
            print(f"Skipping {category}: {exc}")
