from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_hdp_dirs(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "heatmaps": output_dir / "heatmaps",
        "visualizations": output_dir / "visualizations",
        "results": output_dir / "results",
        "logs": output_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def resize_short_edge(image: Image.Image, short_edge: int) -> Image.Image:
    width, height = image.size
    if min(width, height) == short_edge:
        return image
    if width <= height:
        new_width = short_edge
        new_height = int(round(height * short_edge / width))
    else:
        new_height = short_edge
        new_width = int(round(width * short_edge / height))
    return image.resize((new_width, new_height), Image.BICUBIC)


def normalize_array(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    mn = float(np.min(array))
    mx = float(np.max(array))
    return ((array - mn) / (mx - mn + eps)).astype(np.float32)


def topk_mean(array: np.ndarray, ratio: float = 0.002) -> float:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    k = max(1, int(round(flat.size * max(float(ratio), 1e-6))))
    k = min(k, flat.size)
    return float(np.mean(np.partition(flat, flat.size - k)[-k:]))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def numpy_to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def make_relative_safe_name(value: str) -> str:
    return value.replace("\\", "_").replace("/", "_").replace(" ", "_")
