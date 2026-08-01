from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


MAP_REFINE_MODE = "none"
MAP_REFINE_ALPHA = 0.0
MAP_REFINE_BG_SIGMA = 7.0
MAP_REFINE_TOPK_RATIO = 0.03
MAP_REFINE_CLAMP_QUANTILE = 1.0


def refine_anomaly_map(
    anomaly_map: np.ndarray,
    mode: str = "none",
    alpha: float = 0.0,
    bg_sigma: float = 7.0,
    topk_ratio: float = 0.03,
    clamp_quantile: float = 1.0,
) -> np.ndarray:
    """Lightweight localization enhancement for already computed anomaly maps.

    The refinement is intentionally conservative: it keeps the original anomaly
    map as the base and only adds positive local contrast residuals.
    """

    array = np.asarray(anomaly_map, dtype=np.float32)
    alpha = max(float(alpha), 0.0)
    if mode == "none" or alpha <= 0:
        return array

    bg_sigma = max(float(bg_sigma), 0.1)
    background = gaussian_filter(array, sigma=bg_sigma)
    residual = np.maximum(array - background, 0.0).astype(np.float32)

    if mode == "topk_contrast":
        topk_ratio = min(max(float(topk_ratio), 1e-6), 1.0)
        threshold = float(np.quantile(array, 1.0 - topk_ratio))
        residual = np.where(array >= threshold, residual, 0.0).astype(np.float32)
    elif mode != "local_contrast":
        raise ValueError(f"Unsupported map refinement mode: {mode}")

    base_scale = float(np.std(array)) + 1e-6
    residual_scale = float(np.std(residual)) + 1e-6
    enhanced = array + alpha * residual * (base_scale / residual_scale)

    clamp_quantile = float(clamp_quantile)
    if 0.0 < clamp_quantile < 1.0:
        hi = float(np.quantile(enhanced, clamp_quantile))
        enhanced = np.minimum(enhanced, hi)

    return enhanced.astype(np.float32, copy=False)
