"""Semantic-Visual Response Consistency Calibration Module.

This module is parameter-free and uses only existing heatmaps at inference time.
It does not read data, train parameters, or modify the FG-CLIP backbone.
"""

from __future__ import annotations

import numpy as np


def normalize_heatmap(heatmap):
    """Normalize a heatmap to [0, 1] with stable handling for constants."""
    array = np.asarray(heatmap, dtype=np.float32)
    if array.size == 0:
        raise ValueError("heatmap must not be empty")

    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros_like(array, dtype=np.float32)
    if not finite_mask.all():
        finite_min = np.min(array[finite_mask])
        array = np.where(finite_mask, array, finite_min).astype(np.float32)

    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


def _ensure_same_shape(reference: np.ndarray, name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != reference.shape:
        raise ValueError(f"{name} shape mismatch: expected {reference.shape}, got {array.shape}")
    return array


def compute_response_consistency(prompt_heatmap, multiscale_heatmap):
    """Compute semantic-visual response consistency from two heatmaps."""
    prompt = normalize_heatmap(prompt_heatmap)
    multiscale = normalize_heatmap(multiscale_heatmap)
    _ensure_same_shape(prompt, "multiscale_heatmap", multiscale)
    consistency = 1.0 - np.abs(prompt - multiscale)
    return np.clip(consistency, 0.0, 1.0).astype(np.float32)


def calibrate_response(
    base_heatmap,
    prompt_heatmap,
    mask_enhanced_heatmap,
    multiscale_heatmap,
    candidate_mask=None,
    alpha=0.15,
    positive_only=True,
):
    """Calibrate a mask-enhanced heatmap with semantic-visual consistency.

    Parameters are all heatmaps from existing inference stages. The base heatmap
    is normalized and shape-checked to keep the interface explicit, although the
    current calibration formula uses the mask-enhanced heatmap as its anchor.
    """
    base = normalize_heatmap(base_heatmap)
    prompt = normalize_heatmap(prompt_heatmap)
    mask_enhanced = normalize_heatmap(mask_enhanced_heatmap)
    multiscale = normalize_heatmap(multiscale_heatmap)

    _ensure_same_shape(base, "prompt_heatmap", prompt)
    _ensure_same_shape(base, "mask_enhanced_heatmap", mask_enhanced)
    _ensure_same_shape(base, "multiscale_heatmap", multiscale)

    consistency = compute_response_consistency(prompt, multiscale)
    if candidate_mask is None:
        candidate = np.ones_like(base, dtype=np.float32)
    else:
        candidate = _ensure_same_shape(base, "candidate_mask", candidate_mask)
        candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)

    alpha = float(alpha)
    if positive_only:
        delta = np.maximum(multiscale - mask_enhanced, 0.0)
        final = mask_enhanced + alpha * candidate * consistency * delta
    else:
        final = mask_enhanced + alpha * candidate * consistency * multiscale

    final = normalize_heatmap(final)
    if positive_only:
        # Keep the calibration strictly enhancement-only in the normalized space.
        final = np.maximum(final, mask_enhanced)
    return np.clip(final, 0.0, 1.0).astype(np.float32)
