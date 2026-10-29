from __future__ import annotations

import csv
from pathlib import Path

from config import DATASET_ROOT, SELECTED_CATEGORIES, ensure_directories
from data_loader import build_category_dataset, discover_categories, save_dataset_statistics_csv


def scan_category(category: str, dataset_root: Path):
    dataset = build_category_dataset(category=category, dataset_root=dataset_root)
    report = {
        "category": category,
        "train_good_count": len(dataset.train_good),
        "test_good_count": len(dataset.test_good),
        "defective_test_count": sum(len(files) for files in dataset.defect_images.values()),
        "defect_names": sorted(dataset.defect_images.keys()),
        "ground_truth_count": sum(len(files) for files in dataset.ground_truth_masks.values()),
        "ground_truth_present": dataset.ground_truth_present,
        "missing_dirs": dataset.missing_dirs,
        "corrupt_images": dataset.corrupt_images,
    }
    return report


def main() -> None:
    ensure_directories()
    root = DATASET_ROOT
    categories = discover_categories(root)
    print("DATASET ROOT:", root)
    print("AVAILABLE CATEGORIES:", categories)
    print("SELECTED CATEGORIES:", [cat for cat in SELECTED_CATEGORIES if cat in categories])

    rows = []
    for category in SELECTED_CATEGORIES:
        if category not in categories:
            print(f"[{category}] not found in dataset root.")
            rows.append({
                "category": category,
                "status": "missing",
                "train_good_count": 0,
                "test_good_count": 0,
                "defective_test_count": 0,
                "gross_ground_truth_count": 0,
                "defect_names": [],
                "missing_dirs": [],
                "corrupt_images": [],
            })
            continue

        report = scan_category(category, root)
        rows.append({
            "category": report["category"],
            "status": "ok",
            "train_good_count": report["train_good_count"],
            "test_good_count": report["test_good_count"],
            "defective_test_count": report["defective_test_count"],
            "ground_truth_count": report["ground_truth_count"],
            "defect_names": ", ".join(report["defect_names"]) if report["defect_names"] else "-",
            "missing_dirs": ", ".join(report["missing_dirs"]) if report["missing_dirs"] else "-",
            "corrupt_images": ", ".join(report["corrupt_images"]) if report["corrupt_images"] else "-",
        })

        print(f"\n[{category}]")
        print(f"  normal training images: {report['train_good_count']}")
        print(f"  normal test images: {report['test_good_count']}")
        print(f"  defective test images: {report['defective_test_count']}")
        print(f"  defect folders: {report['defect_names'] if report['defect_names'] else 'none'}")
        print(f"  available masks: {report['ground_truth_count']}")
        print(f"  missing directories: {report['missing_dirs'] if report['missing_dirs'] else 'none'}")
        print(f"  corrupt images: {report['corrupt_images'] if report['corrupt_images'] else 'none'}")

    save_dataset_statistics_csv(rows)
    print("\nDataset summary saved to outputs/dataset_statistics/dataset_summary.csv")


if __name__ == "__main__":
    main()
