import math

import numpy as np


def _as_float_array(value):
    array = np.asarray(value, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _clip01(value):
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _safe_mask(candidate_mask, shape):
    if candidate_mask is None:
        return None
    mask = _as_float_array(candidate_mask)
    if mask.shape != shape:
        raise ValueError(f"candidate_mask shape mismatch: {mask.shape} vs {shape}")
    return mask > 0


def normalize_heatmap(heatmap):
    array = _as_float_array(heatmap)
    if array.size == 0:
        return array.astype(np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if not math.isfinite(min_value) or not math.isfinite(max_value) or max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


def compute_quality_components(heatmap, candidate_mask=None, topk_ratio=0.05):
    normalized = normalize_heatmap(heatmap)
    flat = normalized.reshape(-1)
    if flat.size == 0:
        return 0.0, 0.0, 0.0

    ratio = min(max(float(topk_ratio), 1.0 / max(flat.size, 1)), 1.0)
    topk = min(max(1, int(round(flat.size * ratio))), flat.size)
    topk_mean = float(np.mean(np.partition(flat, flat.size - topk)[-topk:]))
    mean_value = float(np.mean(flat))
    focus_score = _clip01(topk_mean - mean_value)
    compactness_score = _clip01(1.0 - mean_value)

    mask = _safe_mask(candidate_mask, normalized.shape)
    if mask is not None:
        overlap_score = float(np.mean(normalized[mask])) if np.any(mask) else 0.0
    else:
        overlap_score = 1.0
    overlap_score = _clip01(overlap_score)
    return focus_score, overlap_score, compactness_score


def compute_quality_score(heatmap, candidate_mask=None, topk_ratio=0.05):
    focus_score, overlap_score, compactness_score = compute_quality_components(
        heatmap,
        candidate_mask=candidate_mask,
        topk_ratio=topk_ratio,
    )
    return float((focus_score + overlap_score + compactness_score) / 3.0)


def compute_risk_components(heatmap, candidate_mask=None, area_q=0.80):
    normalized = normalize_heatmap(heatmap)
    flat = normalized.reshape(-1)
    if flat.size == 0:
        return 0.0, 0.0, 0.0

    global_activation = _clip01(float(np.mean(flat)))
    q = min(max(float(area_q), 0.0), 1.0)
    threshold = float(np.quantile(flat, q))
    area_ratio = _clip01(float(np.mean(flat > threshold)))

    mask = _safe_mask(candidate_mask, normalized.shape)
    if mask is not None and np.any(~mask):
        background_response = float(np.mean(normalized[~mask]))
    else:
        background_response = float(np.mean(flat))
    background_response = _clip01(background_response)
    return global_activation, area_ratio, background_response


def compute_risk_score(heatmap, candidate_mask=None, area_q=0.80):
    global_activation, area_ratio, background_response = compute_risk_components(
        heatmap,
        candidate_mask=candidate_mask,
        area_q=area_q,
    )
    return float((global_activation + area_ratio + background_response) / 3.0)


def calibrate_risk_threshold(
    normal_enhanced_heatmaps,
    candidate_masks=None,
    risk_quantile=0.95,
    area_q=0.80,
):
    heatmaps = list(normal_enhanced_heatmaps)
    if candidate_masks is None:
        masks = [None] * len(heatmaps)
    else:
        masks = list(candidate_masks)
        if len(masks) != len(heatmaps):
            raise ValueError(f"candidate_masks length mismatch: {len(masks)} vs {len(heatmaps)}")

    risk_values = [
        compute_risk_score(heatmap, candidate_mask=mask, area_q=area_q)
        for heatmap, mask in zip(heatmaps, masks)
    ]
    if not risk_values:
        return float("inf")
    q = min(max(float(risk_quantile), 0.0), 1.0)
    return float(np.quantile(np.asarray(risk_values, dtype=np.float32), q))


def select_response(
    baseline_heatmap,
    enhanced_heatmap,
    candidate_mask=None,
    tau_risk=None,
    margin=0.0,
    topk_ratio=0.05,
    area_q=0.80,
):
    q_base = compute_quality_score(baseline_heatmap, candidate_mask=candidate_mask, topk_ratio=topk_ratio)
    q_enh = compute_quality_score(enhanced_heatmap, candidate_mask=candidate_mask, topk_ratio=topk_ratio)
    r_enh = compute_risk_score(enhanced_heatmap, candidate_mask=candidate_mask, area_q=area_q)
    tau_r = float("inf") if tau_risk is None else float(tau_risk)

    if q_enh > q_base + float(margin) and r_enh <= tau_r:
        selected = _as_float_array(enhanced_heatmap)
        selected_source = "enhanced"
    else:
        selected = _as_float_array(baseline_heatmap)
        selected_source = "baseline"

    return selected, selected_source, float(q_base), float(q_enh), float(r_enh), tau_r
