import numpy as np


def normalize_heatmap(heatmap):
    array = np.asarray(heatmap, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.size == 0:
        return array.astype(np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


def compute_response_quality(
    heatmap,
    candidate_mask=None,
    topk_ratio=0.05,
):
    normalized = normalize_heatmap(heatmap)
    flat = normalized.reshape(-1)
    if flat.size == 0:
        return {
            "quality": 0.0,
            "focus": 0.0,
            "compactness": 0.0,
            "overlap": 0.0,
        }

    ratio = min(max(float(topk_ratio), 1.0 / max(flat.size, 1)), 1.0)
    topk = max(1, int(round(flat.size * ratio)))
    topk = min(topk, flat.size)
    topk_mean = float(np.mean(np.partition(flat, flat.size - topk)[-topk:]))
    global_mean = float(np.mean(flat))
    focus = max(topk_mean - global_mean, 0.0)
    compactness = 1.0 - global_mean

    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=np.float32)
        if mask.shape != normalized.shape:
            raise ValueError(f"candidate_mask shape mismatch: {mask.shape} vs {normalized.shape}")
        positive = mask > 0
        if np.any(positive):
            overlap = float(np.mean(normalized[positive]))
        else:
            overlap = 0.0
    else:
        overlap = 1.0

    quality = float(0.5 * focus + 0.3 * overlap + 0.2 * compactness)
    return {
        "quality": quality,
        "focus": float(focus),
        "compactness": float(compactness),
        "overlap": float(overlap),
    }


def safe_select_response(
    baseline_heatmap,
    enhanced_heatmap,
    candidate_mask=None,
    margin=0.02,
    topk_ratio=0.05,
    enhanced_bias=0.0,
):
    q_base_stats = compute_response_quality(
        baseline_heatmap,
        candidate_mask=candidate_mask,
        topk_ratio=topk_ratio,
    )
    q_enh_stats = compute_response_quality(
        enhanced_heatmap,
        candidate_mask=candidate_mask,
        topk_ratio=topk_ratio,
    )

    q_base = float(q_base_stats["quality"])
    q_enh = float(q_enh_stats["quality"])

    if q_enh + float(enhanced_bias) > q_base + float(margin):
        selected = np.asarray(enhanced_heatmap, dtype=np.float32)
        selected_source = "enhanced"
    else:
        selected = np.asarray(baseline_heatmap, dtype=np.float32)
        selected_source = "baseline"
    return selected, selected_source, q_base, q_enh, q_base_stats, q_enh_stats
