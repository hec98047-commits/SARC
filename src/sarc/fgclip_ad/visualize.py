from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .utils import normalize_array


def colorize_heatmap(anomaly_map: np.ndarray) -> Image.Image:
    norm = normalize_array(anomaly_map)
    r = np.clip(2.0 * norm, 0, 1)
    g = np.clip(2.0 * (1.0 - np.abs(norm - 0.5)), 0, 1)
    b = np.clip(2.0 * (1.0 - norm), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def save_heatmap_overlay(image: Image.Image, anomaly_map: np.ndarray, path: Path, alpha: float = 0.45) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = colorize_heatmap(anomaly_map).resize(image.size, Image.BICUBIC)
    overlay = Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha)
    overlay.save(path)


def save_heatmap_tiff(anomaly_map: np.ndarray, path: Path) -> None:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(anomaly_map, dtype=np.float32))
