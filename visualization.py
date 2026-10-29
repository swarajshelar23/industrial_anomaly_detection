from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ANOMALY_HEATMAPS_DIR, CONFUSION_MATRIX_DIR, MODEL_COMPARISON_DIR, PLOTS_DIR, ROC_CURVES_DIR, TRAINING_CURVES_DIR


def save_training_history_plot(history_path: str | Path, output_path: str | Path):
    df = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curve(labels: Sequence[int], scores: Sequence[float], output_path: str | Path):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_plot(labels: Sequence[int], predictions: Sequence[int], output_path: str | Path):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomalous"])
    ax.set_yticklabels(["Normal", "Anomalous"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model_comparison_table(summary_csv: str | Path, output_dir: str | Path):
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for metric, axis, title in [
        ("roc_auc", axes[0], "ROC-AUC by Category and Model"),
        ("f1", axes[1], "F1-score by Category and Model"),
    ]:
        pivot = df.pivot(index="category", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(title)
        axis.set_ylabel(metric.upper() if metric == "roc_auc" else "F1")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_summary.png", dpi=200)
    plt.close(fig)


def save_metric_heatmaps(summary_csv: str | Path, output_dir: str | Path):
    df = pd.read_csv(summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["roc_auc", "precision", "recall", "f1", "accuracy", "fpr"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for metric, axis in zip(metrics, axes.flat):
        table = df.pivot(index="category", columns="model", values=metric)
        image = axis.imshow(table.values, cmap="viridis", vmin=0, vmax=1)
        axis.set_title(metric.upper())
        axis.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
        axis.set_yticks(range(len(table.index)), table.index)
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                axis.text(col, row, f"{table.iloc[row, col]:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_heatmaps.png", dpi=200)
    plt.close(fig)


def save_confusion_matrices_from_summary(summary_csv: str | Path, output_dir: str | Path):
    df = pd.read_csv(summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(cm, cmap="Blues")
        axis.set_title(f'{row["category"].title()} - {row["model"]}')
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1], ["Normal", "Anomalous"])
        axis.set_yticks([0, 1], ["Normal", "Anomalous"])
        for i in range(2):
            for j in range(2):
                axis.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(output_dir / f'{row["category"]}_{row["model"]}_confusion_matrix.png', dpi=200)
        plt.close(fig)


def save_all_training_curves(histories_dir: str | Path, output_dir: str | Path):
    histories_dir = Path(histories_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for history_path in sorted(histories_dir.glob("*_history.csv")):
        save_training_history_plot(history_path, output_dir / f"{history_path.stem}.png")


if __name__ == "__main__":
    from config import HISTORIES_DIR, METRICS_DIR

    summary_path = METRICS_DIR / "model_comparison_summary.csv"
    save_all_training_curves(HISTORIES_DIR, TRAINING_CURVES_DIR)
    save_model_comparison_table(summary_path, MODEL_COMPARISON_DIR)
    save_metric_heatmaps(summary_path, MODEL_COMPARISON_DIR)
    save_confusion_matrices_from_summary(summary_path, CONFUSION_MATRIX_DIR)
    print(f"Plots saved under {PLOTS_DIR}")
