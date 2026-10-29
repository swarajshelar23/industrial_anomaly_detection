from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError

from config import DATASET_ROOT, DATASET_STATS_DIR, SELECTED_CATEGORIES

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class CategoryDataset:
    category: str
    category_path: Path
    train_good: List[Path] = field(default_factory=list)
    test_good: List[Path] = field(default_factory=list)
    defect_images: Dict[str, List[Path]] = field(default_factory=dict)
    ground_truth_masks: Dict[str, List[Path]] = field(default_factory=dict)
    defect_names: List[str] = field(default_factory=list)
    missing_dirs: List[str] = field(default_factory=list)
    corrupt_images: List[str] = field(default_factory=list)
    ground_truth_present: bool = False

    @property
    def train_count(self) -> int:
        return len(self.train_good)

    @property
    def normal_test_count(self) -> int:
        return len(self.test_good)

    @property
    def anomalous_test_count(self) -> int:
        return sum(len(files) for files in self.defect_images.values())


def discover_categories(dataset_root: Path | str | None = None) -> List[str]:
    root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    categories = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.lower() in SELECTED_CATEGORIES:
            categories.append(child.name.lower())
    return categories


def _list_image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return False


def _collect_corrupt_images(paths: List[Path]) -> List[str]:
    corrupt = []
    for path in paths:
        if not _is_valid_image(path):
            corrupt.append(str(path.name))
    return corrupt


def build_category_dataset(category: str, dataset_root: Path | str | None = None) -> CategoryDataset:
    root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
    category_path = root / category
    if not category_path.exists():
        raise FileNotFoundError(f"Category folder not found: {category_path}")

    train_dir = category_path / "train" / "good"
    test_dir = category_path / "test"
    ground_truth_dir = category_path / "ground_truth"

    missing_dirs = []
    if not train_dir.exists():
        missing_dirs.append(str(train_dir.relative_to(root)))
    if not test_dir.exists():
        missing_dirs.append(str(test_dir.relative_to(root)))
    if not ground_truth_dir.exists():
        missing_dirs.append(str(ground_truth_dir.relative_to(root)))

    train_good = _list_image_files(train_dir)
    test_good = _list_image_files(test_dir / "good") if (test_dir / "good").exists() else []

    defect_images: Dict[str, List[Path]] = {}
    ground_truth_masks: Dict[str, List[Path]] = {}

    if test_dir.exists():
        for child in sorted(test_dir.iterdir()):
            if child.is_dir() and child.name != "good":
                defect_name = child.name
                defect_images[defect_name] = _list_image_files(child)
                mask_dir = ground_truth_dir / defect_name
                ground_truth_masks[defect_name] = _list_image_files(mask_dir) if mask_dir.exists() else []

    defect_names = sorted(defect_images)
    corrupt_images = _collect_corrupt_images(train_good + test_good + [p for files in defect_images.values() for p in files])

    categories_with_masks = sum(1 for files in ground_truth_masks.values() if files)
    ground_truth_present = categories_with_masks > 0 or any((ground_truth_dir / defect).exists() for defect in defect_names)

    return CategoryDataset(
        category=category,
        category_path=category_path,
        train_good=train_good,
        test_good=test_good,
        defect_images=defect_images,
        ground_truth_masks=ground_truth_masks,
        defect_names=defect_names,
        missing_dirs=missing_dirs,
        corrupt_images=corrupt_images,
        ground_truth_present=ground_truth_present,
    )


def save_dataset_statistics_csv(rows: List[dict], output_path: Path | str | None = None) -> Path:
    """Save dataset statistics in CSV format."""
    output_dir = Path(output_path) if output_path is not None else DATASET_STATS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_summary.csv"

    fieldnames = [
        "category",
        "status",
        "train_good_count",
        "test_good_count",
        "defective_test_count",
        "ground_truth_count",
        "defect_names",
        "missing_dirs",
        "corrupt_images",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def build_all_datasets(dataset_root: Path | str | None = None) -> Dict[str, CategoryDataset]:
    root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
    datasets = {}
    for category in discover_categories(root):
        datasets[category] = build_category_dataset(category=category, dataset_root=root)
    return datasets


if __name__ == "__main__":
    datasets = build_all_datasets(DATASET_ROOT)
    for name, dataset in datasets.items():
        print(name, len(dataset.train_good), len(dataset.test_good), sum(len(v) for v in dataset.defect_images.values()))
