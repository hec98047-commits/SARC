from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import list_images


MVTEC_OBJECTS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass
class AnomalySample:
    dataset: str
    class_name: str
    split: str
    image_path: Path
    label_name: str
    mask_path: Path | None = None

    @property
    def is_anomaly(self) -> bool:
        return self.label_name not in {"good", "normal"}


def discover_classes(dataset: str, data_root: Path, split_csv: Path | None = None) -> list[str]:
    data_root = Path(data_root)
    if dataset == "mvtec":
        return [name for name in MVTEC_OBJECTS if (data_root / name).is_dir()]
    csv_path = split_csv or data_root / "split_csv" / "1cls.csv"
    if csv_path.exists():
        entries = load_visa_split(csv_path, data_root)
        return sorted(entries.keys())
    return sorted([path.name for path in data_root.iterdir() if path.is_dir()])


def load_mvtec_samples(data_root: Path, class_name: str) -> tuple[list[AnomalySample], list[AnomalySample]]:
    data_root = Path(data_root)
    train_good_dir = data_root / class_name / "train" / "good"
    test_dir = data_root / class_name / "test"

    train_samples = [
        AnomalySample("mvtec", class_name, "train", path, "good", None)
        for path in list_images(train_good_dir)
    ]

    test_samples: list[AnomalySample] = []
    for image_path in list_images(test_dir):
        defect_type = image_path.parent.name
        mask_path = None
        if defect_type != "good":
            mask_path = data_root / class_name / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
        test_samples.append(AnomalySample("mvtec", class_name, "test", image_path, defect_type, mask_path))
    return train_samples, sorted(test_samples, key=lambda item: str(item.image_path))


def load_visa_split(split_csv: Path, data_root: Path) -> dict[str, dict[str, list[AnomalySample]]]:
    split_csv = Path(split_csv)
    data_root = Path(data_root)
    if not split_csv.exists():
        raise FileNotFoundError(f"VisA split csv not found: {split_csv}")

    entries: dict[str, dict[str, list[AnomalySample]]] = {}
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["object"]
            split = row["split"]
            label_name = row["label"]
            mask_path = data_root / row["mask"] if row.get("mask") else None
            sample = AnomalySample(
                dataset="visa",
                class_name=class_name,
                split=split,
                image_path=data_root / row["image"],
                label_name=label_name,
                mask_path=mask_path,
            )
            entries.setdefault(class_name, {}).setdefault(split, []).append(sample)
    return entries


def load_visa_samples_from_folders(data_root: Path, class_name: str) -> tuple[list[AnomalySample], list[AnomalySample]]:
    data_root = Path(data_root)
    class_root = data_root / class_name
    train_good_dir = class_root / "train" / "good"
    test_dir = class_root / "test"
    gt_dir = class_root / "ground_truth"

    train_samples = [
        AnomalySample("visa", class_name, "train", path, "good", None)
        for path in list_images(train_good_dir)
    ]

    test_samples: list[AnomalySample] = []
    for image_path in list_images(test_dir):
        label_name = image_path.parent.name
        mask_path = None
        if label_name not in {"good", "normal"}:
            candidate_masks = [
                gt_dir / label_name / f"{image_path.stem}.png",
                gt_dir / label_name / f"{image_path.stem}.jpg",
                gt_dir / label_name / f"{image_path.stem}_mask.png",
            ]
            for candidate in candidate_masks:
                if candidate.exists():
                    mask_path = candidate
                    break
        test_samples.append(AnomalySample("visa", class_name, "test", image_path, label_name, mask_path))
    return train_samples, sorted(test_samples, key=lambda item: str(item.image_path))


def load_visa_samples(data_root: Path, class_name: str, split_csv: Path | None = None):
    csv_path = split_csv or Path(data_root) / "split_csv" / "1cls.csv"
    if csv_path.exists():
        entries = load_visa_split(csv_path, data_root)
        class_entries = entries[class_name]
        train_samples = [item for item in class_entries.get("train", []) if not item.is_anomaly]
        test_samples = class_entries.get("test", [])
        return train_samples, sorted(test_samples, key=lambda item: str(item.image_path))
    return load_visa_samples_from_folders(data_root, class_name)


def load_dataset_samples(
    dataset: str,
    data_root: Path,
    class_name: str,
    split_csv: Path | None = None,
) -> tuple[list[AnomalySample], list[AnomalySample]]:
    if dataset == "mvtec":
        return load_mvtec_samples(data_root, class_name)
    if dataset == "visa":
        return load_visa_samples(data_root, class_name, split_csv)
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_mask(sample: AnomalySample, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if sample.mask_path is None or not sample.mask_path.exists():
        return np.zeros((height, width), dtype=np.uint8)
    mask = Image.open(sample.mask_path)
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)
