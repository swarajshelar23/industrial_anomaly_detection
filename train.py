from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from anomaly_detection import compute_threshold_from_validation, detect_anomalies_for_category
from config import (
    BATCH_SIZE,
    DATASET_ROOT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HISTORIES_DIR,
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
    QUICK_TEST_EPOCHS,
    QUICK_TEST_MAX_TRAIN_SAMPLES,
    QUICK_TEST_MAX_VAL_SAMPLES,
    RANDOM_SEED,
    SELECTED_CATEGORIES,
    ensure_directories,
)
from data_loader import build_category_dataset, discover_categories
from models import model_name_to_builder
from preprocessing import build_image_dataset, prepare_train_val_split, set_global_seed

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoencoder models for MVTec anomaly detection.")
    parser.add_argument("--category", choices=SELECTED_CATEGORIES, required=True, help="Category to train.")
    parser.add_argument("--model", choices=["ae_cnn", "denoising_ae_cnn", "vae_cnn"], required=True, help="Model architecture to train.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT), help="Root directory of the MVTec AD dataset.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Image size for resizing.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--quick-test", action="store_true", help="Use a minimal data subset for a quick pipeline verification.")
    parser.add_argument("--no-resume", action="store_true", help="Force retraining even if a saved model exists.")
    return parser.parse_args()


def save_history_csv(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        epochs = len(history.history.get("loss", []))
        for idx in range(epochs):
            train_loss = history.history.get("loss", [None])[idx]
            val_loss = history.history.get("val_loss", [None])[idx]
            writer.writerow([idx + 1, train_loss, val_loss])


def train_category_model(category: str, model_name: str, dataset_root: str, image_size: int = IMAGE_SIZE, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS, patience: int = EARLY_STOPPING_PATIENCE, quick_test: bool = False, force_retrain: bool = False):
    ensure_directories()
    set_global_seed(RANDOM_SEED)
    root_path = Path(dataset_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    dataset = build_category_dataset(category=category, dataset_root=root_path)
    if not dataset.train_good:
        raise ValueError(f"No training images found for category '{category}' in {root_path}")

    train_paths, val_paths = prepare_train_val_split([str(p) for p in dataset.train_good], val_fraction=0.2, seed=RANDOM_SEED)
    if quick_test:
        train_paths = train_paths[:QUICK_TEST_MAX_TRAIN_SAMPLES]
        val_paths = val_paths[:QUICK_TEST_MAX_VAL_SAMPLES] if val_paths else train_paths[:QUICK_TEST_MAX_VAL_SAMPLES]
        epochs = QUICK_TEST_EPOCHS

    model_builder = model_name_to_builder(model_name)
    model = model_builder(input_shape=(image_size, image_size, 3))

    model_path = MODELS_DIR / f"{category}_{model_name}.keras"
    history_path = HISTORIES_DIR / f"{category}_{model_name}_history.csv"

    if model_path.exists() and not force_retrain:
        print(f"Model already exists: {model_path}. Use --no-resume to retrain.")
        return {
            "category": category,
            "model": model_name,
            "model_path": str(model_path),
            "history_path": str(history_path),
            "skipped": True,
        }

    train_dataset = build_image_dataset(train_paths, batch_size=batch_size, image_size=image_size, shuffle=True, seed=RANDOM_SEED, add_noise=(model_name == "denoising_ae_cnn"), noise_std=0.1)
    val_dataset = build_image_dataset(val_paths, batch_size=batch_size, image_size=image_size, shuffle=False, seed=RANDOM_SEED, add_noise=False)

    if model_name == "vae_cnn":
        val_dataset = build_image_dataset(val_paths, batch_size=batch_size, image_size=image_size, shuffle=False, seed=RANDOM_SEED, add_noise=False)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1,
    )
    elapsed = time.time() - start_time

    save_history_csv(history, history_path)

    result = {
        "category": category,
        "model": model_name,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "training_time_seconds": round(elapsed, 3),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "quick_test": quick_test,
        "skipped": False,
    }
    return result


if __name__ == "__main__":
    args = parse_args()
    train_category_model(
        category=args.category,
        model_name=args.model,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        quick_test=args.quick_test,
        force_retrain=args.no_resume,
    )
