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


def compute_reliability_weight(
    tile_heatmap,
    candidate_mask=None,
    topk_ratio=0.05,
    min_weight=0.0,
    max_weight=1.0,
):
    tile = normalize_heatmap(tile_heatmap)
    flat = tile.reshape(-1)
    if flat.size == 0:
        return {
            "weight": 0.0,
            "R_focus": 0.0,
            "R_spread": 0.0,
            "R_overlap": 0.0,
        }

    ratio = min(max(float(topk_ratio), 1.0 / max(flat.size, 1)), 1.0)
    topk = max(1, int(round(flat.size * ratio)))
    topk = min(topk, flat.size)
    topk_mean = float(np.mean(np.partition(flat, flat.size - topk)[-topk:]))
    global_mean = float(np.mean(flat))

    r_focus = max(topk_mean - global_mean, 0.0)
    r_spread = 1.0 - global_mean

    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=np.float32)
        if mask.shape != tile.shape:
            raise ValueError(f"candidate_mask shape mismatch: {mask.shape} vs {tile.shape}")
        positive = mask > 0
        if np.any(positive):
            r_overlap = float(np.mean(tile[positive]))
        else:
            r_overlap = 0.0
    else:
        r_overlap = 1.0

    reliability = r_focus * r_spread * r_overlap
    reliability = float(np.clip(reliability, float(min_weight), float(max_weight)))
    return {
        "weight": reliability,
        "R_focus": float(r_focus),
        "R_spread": float(r_spread),
        "R_overlap": float(r_overlap),
    }


def reliability_aware_positive_fusion(
    mask_heatmap,
    tile_heatmap,
    candidate_mask=None,
    beta=0.08,
    topk_ratio=0.05,
    min_weight=0.0,
    max_weight=1.0,
    return_details=False,
):
    mask_map = normalize_heatmap(mask_heatmap)
    tile_map = normalize_heatmap(tile_heatmap)
    if mask_map.shape != tile_map.shape:
        raise ValueError(f"heatmap shape mismatch: {mask_map.shape} vs {tile_map.shape}")

    stats = compute_reliability_weight(
        tile_map,
        candidate_mask=candidate_mask,
        topk_ratio=topk_ratio,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    delta = np.maximum(tile_map - mask_map, 0.0)

    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=np.float32)
        if mask.shape != mask_map.shape:
            raise ValueError(f"candidate_mask shape mismatch: {mask.shape} vs {mask_map.shape}")
        final = mask_map + float(beta) * stats["weight"] * mask * delta
    else:
        final = mask_map + float(beta) * stats["weight"] * delta

    final = normalize_heatmap(final)
    if return_details:
        return final, stats
    return final
