from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .utils import normalize_array
from .visualize import colorize_heatmap, save_heatmap_overlay


EPSILON = 1e-8


def ensure_finite_array(name: str, array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return value


def ensure_finite_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return value


def l2_normalize_tensor(tensor: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(eps)


def sigmoid_array(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(array, dtype=np.float32), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def robust_normalize_heatmap(
    heatmap: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = EPSILON,
) -> np.ndarray:
    value = ensure_finite_array("heatmap", heatmap)
    low = float(np.percentile(value, p_low))
    high = float(np.percentile(value, p_high))
    denom = high - low
    if not np.isfinite(denom) or denom < eps:
        low = float(np.min(value))
        high = float(np.max(value))
        denom = high - low
    if not np.isfinite(denom) or denom < eps:
        return np.zeros_like(value, dtype=np.float32)
    normalized = (value - low) / (denom + float(eps))
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    return ensure_finite_array("robust_normalize_heatmap", normalized)


def infer_patch_grid_shape(num_patches: int) -> tuple[int, int]:
    side = int(round(float(num_patches) ** 0.5))
    if side * side != int(num_patches):
        raise ValueError(
            "patch_grid_shape is required when patch scores do not form a square grid. "
            f"Received num_patches={num_patches}."
        )
    return side, side


def compute_text_conditioned_score(
    patch_features: torch.Tensor,
    normal_text_features: torch.Tensor,
    abnormal_text_features: torch.Tensor,
) -> torch.Tensor:
    if patch_features.ndim != 2:
        raise ValueError(f"patch_features must be [N, D], got {tuple(patch_features.shape)}")
    if normal_text_features.ndim == 1:
        normal_text_features = normal_text_features.unsqueeze(0)
    if abnormal_text_features.ndim == 1:
        abnormal_text_features = abnormal_text_features.unsqueeze(0)
    if normal_text_features.ndim != 2 or abnormal_text_features.ndim != 2:
        raise ValueError("text features must be [K, D] or [D].")
    if patch_features.shape[-1] != normal_text_features.shape[-1] or patch_features.shape[-1] != abnormal_text_features.shape[-1]:
        raise ValueError(
            "Feature dimension mismatch: "
            f"patch={patch_features.shape[-1]}, "
            f"normal={normal_text_features.shape[-1]}, "
            f"abnormal={abnormal_text_features.shape[-1]}"
        )

    patch_features = l2_normalize_tensor(patch_features.float())
    normal_text_features = l2_normalize_tensor(normal_text_features.float())
    abnormal_text_features = l2_normalize_tensor(abnormal_text_features.float())

    abnormal_sim = patch_features @ abnormal_text_features.T
    normal_sim = patch_features @ normal_text_features.T
    text_score = abnormal_sim.mean(dim=1) - normal_sim.mean(dim=1)
    return ensure_finite_tensor("text_score", text_score)


def patch_score_to_heatmap(
    text_score: torch.Tensor | np.ndarray,
    image_size: tuple[int, int],
    patch_grid_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    score_array = np.asarray(text_score, dtype=np.float32).reshape(-1)
    ensure_finite_array("text_score", score_array)
    if patch_grid_shape is None:
        patch_grid_shape = infer_patch_grid_shape(score_array.size)
    grid_h, grid_w = int(patch_grid_shape[0]), int(patch_grid_shape[1])
    if grid_h * grid_w != score_array.size:
        raise ValueError(
            f"patch score length mismatch: len={score_array.size}, grid={patch_grid_shape}"
        )
    patch_map = score_array.reshape(grid_h, grid_w)
    heatmap = np.asarray(
        Image.fromarray(patch_map, mode="F").resize(tuple(image_size), Image.BICUBIC),
        dtype=np.float32,
    )
    return ensure_finite_array("text_conditioned_heatmap", heatmap)


def compute_text_conditioned_mask(
    text_conditioned_heatmap: np.ndarray,
    tau: float,
    beta: float,
    eps: float = EPSILON,
) -> np.ndarray:
    heatmap = ensure_finite_array("text_conditioned_heatmap", text_conditioned_heatmap)
    normalized = normalize_array(heatmap, eps=eps)
    beta = max(float(beta), eps)
    mask = sigmoid_array((normalized - float(tau)) / beta)
    return ensure_finite_array("text_mask", mask)


def apply_text_conditioned_psme(
    text_conditioned_heatmap: np.ndarray,
    text_mask: np.ndarray,
    lambda_weight: float,
) -> np.ndarray:
    heatmap = ensure_finite_array("text_conditioned_heatmap", text_conditioned_heatmap)
    mask = ensure_finite_array("text_mask", text_mask)
    if heatmap.shape != mask.shape:
        raise ValueError(f"text heatmap/mask shape mismatch: {heatmap.shape} vs {mask.shape}")
    psme_heatmap = heatmap + float(lambda_weight) * mask * np.maximum(heatmap, 0.0)
    return ensure_finite_array("psme_heatmap", psme_heatmap.astype(np.float32))


def _coerce_tile_box(tile_box) -> tuple[int, int, int, int]:
    if isinstance(tile_box, dict):
        return int(tile_box["x1"]), int(tile_box["y1"]), int(tile_box["x2"]), int(tile_box["y2"])
    if len(tile_box) != 4:
        raise ValueError(f"tile_box must have four values, got {tile_box}")
    x1, y1, x2, y2 = tile_box
    return int(x1), int(y1), int(x2), int(y2)


def compute_tile_guidance_scores(
    global_heatmap: np.ndarray,
    tile_boxes: Iterable,
    topk_ratio: float,
) -> np.ndarray:
    heatmap = ensure_finite_array("global_heatmap", global_heatmap)
    ratio = min(max(float(topk_ratio), EPSILON), 1.0)
    scores = []
    for tile_index, tile_box in enumerate(tile_boxes):
        x1, y1, x2, y2 = _coerce_tile_box(tile_box)
        tile_region = heatmap[y1:y2, x1:x2]
        if tile_region.size == 0:
            raise ValueError(f"tile_box[{tile_index}] produced an empty crop: {(x1, y1, x2, y2)}")
        flat = tile_region.reshape(-1)
        k = max(1, int(round(flat.size * ratio)))
        k = min(k, flat.size)
        score = float(np.mean(np.partition(flat, flat.size - k)[-k:]))
        scores.append(score)
    return ensure_finite_array("tile_guidance_scores", np.asarray(scores, dtype=np.float32))


def compute_tile_weights(tile_scores: np.ndarray, temperature: float, eps: float = EPSILON) -> np.ndarray:
    scores = ensure_finite_array("tile_guidance_scores", tile_scores).reshape(-1)
    if scores.size == 0:
        raise ValueError("tile_scores must not be empty.")
    scaled = scores / max(float(temperature), eps)
    scaled = scaled - float(np.max(scaled))
    weights = np.exp(np.clip(scaled, -60.0, 60.0))
    weights = weights / max(float(np.sum(weights)), eps)
    return ensure_finite_array("tile_weights", weights.astype(np.float32))


def restore_tile_heatmaps(
    tile_heatmaps: Iterable[np.ndarray],
    tile_boxes: Iterable,
    image_size: tuple[int, int],
    tile_weights: np.ndarray | None = None,
    eps: float = EPSILON,
) -> np.ndarray:
    image_w, image_h = int(image_size[0]), int(image_size[1])
    weighted_sum = np.zeros((image_h, image_w), dtype=np.float32)
    weight_sum = np.zeros((image_h, image_w), dtype=np.float32)
    tile_heatmaps = list(tile_heatmaps)
    tile_boxes = list(tile_boxes)
    if len(tile_heatmaps) != len(tile_boxes):
        raise ValueError(f"tile count mismatch: heatmaps={len(tile_heatmaps)}, boxes={len(tile_boxes)}")
    if tile_weights is None:
        tile_weights = np.ones(len(tile_heatmaps), dtype=np.float32)
    tile_weights = ensure_finite_array("tile_weights", tile_weights).reshape(-1)
    if tile_weights.size != len(tile_heatmaps):
        raise ValueError(f"tile weight mismatch: weights={tile_weights.size}, heatmaps={len(tile_heatmaps)}")

    for tile_index, (tile_heatmap, tile_box, tile_weight) in enumerate(zip(tile_heatmaps, tile_boxes, tile_weights)):
        heatmap = ensure_finite_array(f"tile_heatmap[{tile_index}]", tile_heatmap)
        x1, y1, x2, y2 = _coerce_tile_box(tile_box)
        resized = np.asarray(
            Image.fromarray(heatmap, mode="F").resize((x2 - x1, y2 - y1), Image.BICUBIC),
            dtype=np.float32,
        )
        resized = robust_normalize_heatmap(resized)
        weighted_sum[y1:y2, x1:x2] += float(tile_weight) * resized
        weight_sum[y1:y2, x1:x2] += float(tile_weight)

    restored = np.divide(weighted_sum, weight_sum + float(eps), out=np.zeros_like(weighted_sum), where=weight_sum > 0)
    restored = robust_normalize_heatmap(restored)
    return ensure_finite_array("local_weighted_heatmap", restored)


def global_guided_multiscale_fusion(
    global_heatmap: np.ndarray,
    tile_heatmaps: Iterable[np.ndarray],
    tile_boxes: Iterable,
    alpha: float,
    topk_ratio: float,
    temperature: float,
) -> dict[str, np.ndarray]:
    heatmap = ensure_finite_array("global_heatmap", global_heatmap)
    tile_boxes = list(tile_boxes)
    global_heatmap_norm = robust_normalize_heatmap(heatmap)
    tile_scores_raw = compute_tile_guidance_scores(heatmap, tile_boxes, topk_ratio=topk_ratio)
    tile_scores = compute_tile_guidance_scores(global_heatmap_norm, tile_boxes, topk_ratio=topk_ratio)
    tile_weights = compute_tile_weights(tile_scores, temperature=temperature)
    local_heatmap = restore_tile_heatmaps(
        tile_heatmaps=tile_heatmaps,
        tile_boxes=tile_boxes,
        image_size=(heatmap.shape[1], heatmap.shape[0]),
        tile_weights=tile_weights,
    )
    local_heatmap_norm = robust_normalize_heatmap(local_heatmap)
    final_heatmap = (1.0 - float(alpha)) * global_heatmap_norm + float(alpha) * local_heatmap_norm
    final_heatmap = ensure_finite_array("final_heatmap", final_heatmap.astype(np.float32))
    return {
        "tile_scores_raw": tile_scores_raw,
        "tile_scores": tile_scores,
        "tile_weights": tile_weights,
        "local_weighted_heatmap": local_heatmap,
        "global_heatmap_norm": global_heatmap_norm,
        "local_heatmap_norm": local_heatmap_norm,
        "final_heatmap": final_heatmap,
    }


def save_intermediate_heatmap(
    output_dir: Path,
    stage_name: str,
    sample_stem: str,
    heatmap: np.ndarray,
    image: Image.Image | None = None,
) -> None:
    heatmap = ensure_finite_array(stage_name, heatmap)
    stage_dir = Path(output_dir) / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.save(stage_dir / f"{sample_stem}.npy", heatmap.astype(np.float32))
    png = colorize_heatmap(heatmap)
    png.save(stage_dir / f"{sample_stem}.png")
    if image is not None:
        overlay_dir = Path(output_dir) / "overlay" / stage_name
        save_heatmap_overlay(image, heatmap, overlay_dir / f"{sample_stem}.png")
