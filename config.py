from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Configure this to your local MVTec AD root, e.g. r"D:\Datasets\mvtec_anomaly_detection"
DATASET_ROOT = Path(r"c:\Users\SUPER\Downloads\industrial_anomaly_detection")
if not DATASET_ROOT.exists():
    DATASET_ROOT = PROJECT_ROOT

SELECTED_CATEGORIES = ["bottle", "cable", "capsule", "hazelnut"]
MODEL_NAMES = ["ae_cnn", "denoising_ae_cnn", "vae_cnn"]

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42
VAL_FRACTION = 0.2
THRESHOLD_PERCENTILE = 95.0
NOISE_STD = 0.1
BETA = 0.1

QUICK_TEST_MAX_TRAIN_SAMPLES = 12
QUICK_TEST_MAX_VAL_SAMPLES = 4
QUICK_TEST_EPOCHS = 3

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_STATS_DIR = OUTPUTS_DIR / "dataset_statistics"
MODELS_DIR = OUTPUTS_DIR / "models"
HISTORIES_DIR = OUTPUTS_DIR / "histories"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

TRAINING_CURVES_DIR = PLOTS_DIR / "training_curves"
ROC_CURVES_DIR = PLOTS_DIR / "roc_curves"
CONFUSION_MATRIX_DIR = PLOTS_DIR / "confusion_matrices"
MODEL_COMPARISON_DIR = PLOTS_DIR / "model_comparison"
ANOMALY_HEATMAPS_DIR = PLOTS_DIR / "anomaly_heatmaps"

def ensure_directories() -> None:
    """Create all directories required for training and output artifacts."""
    for directory in [
        DATASET_STATS_DIR,
        MODELS_DIR,
        HISTORIES_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        TRAINING_CURVES_DIR,
        ROC_CURVES_DIR,
        CONFUSION_MATRIX_DIR,
        MODEL_COMPARISON_DIR,
        ANOMALY_HEATMAPS_DIR,
        PREDICTIONS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
