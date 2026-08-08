from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_components


def binary_auroc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    num_pos = int(np.sum(labels == 1))
    num_neg = int(np.sum(labels == 0))
    if num_pos == 0 or num_neg == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct_mask = np.r_[np.diff(sorted_scores) != 0, True]
    tp = np.cumsum(sorted_labels == 1)[distinct_mask]
    fp = np.cumsum(sorted_labels == 0)[distinct_mask]
    tpr = np.r_[0.0, tp / num_pos, 1.0]
    fpr = np.r_[0.0, fp / num_neg, 1.0]
    return float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])))


def f1_max(scores, labels) -> dict:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    num_pos = int(np.sum(labels == 1))
    if num_pos == 0:
        return {"f1_max": 0.0, "best_threshold": None, "best_precision": 0.0, "best_recall": 0.0}
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct_mask = np.r_[np.diff(sorted_scores) != 0, True]
    tp = np.cumsum(sorted_labels == 1)[distinct_mask].astype(np.float64)
    fp = np.cumsum(sorted_labels == 0)[distinct_mask].astype(np.float64)
    thresholds = sorted_scores[distinct_mask]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / num_pos
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    idx = int(np.argmax(f1))
    return {
        "f1_max": float(f1[idx]),
        "best_threshold": float(thresholds[idx]),
        "best_precision": float(precision[idx]),
        "best_recall": float(recall[idx]),
    }


def pro_auc(anomaly_maps, masks, fpr_limit: float = 0.3) -> float:
    if not anomaly_maps:
        return float("nan")
    structure = np.ones((3, 3), dtype=int)
    scores = []
    fp_changes = []
    pro_changes = []
    num_ok = 0
    num_regions = 0

    for pred, mask in zip(anomaly_maps, masks):
        pred = np.asarray(pred, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.uint8)
        labeled, n_components = connected_components(mask > 0, structure)
        ok_mask = labeled == 0
        num_ok += int(np.sum(ok_mask))
        num_regions += int(n_components)
        fp = np.zeros_like(pred, dtype=np.float32)
        fp[ok_mask] = 1.0
        pro = np.zeros_like(pred, dtype=np.float32)
        for component_idx in range(1, n_components + 1):
            region = labeled == component_idx
            pro[region] = 1.0 / max(int(np.sum(region)), 1)
        scores.append(pred.reshape(-1))
        fp_changes.append(fp.reshape(-1))
        pro_changes.append(pro.reshape(-1))

    if num_ok == 0 or num_regions == 0:
        return float("nan")

    scores = np.concatenate(scores)
    fp_changes = np.concatenate(fp_changes)
    pro_changes = np.concatenate(pro_changes)
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    fp_changes = np.cumsum(fp_changes[order])
    pro_changes = np.cumsum(pro_changes[order])
    fprs = np.clip(fp_changes / num_ok, 0.0, 1.0)
    pros = np.clip(pro_changes / num_regions, 0.0, 1.0)
    keep = np.r_[np.diff(scores) != 0, True]
    fprs = np.r_[0.0, fprs[keep], 1.0]
    pros = np.r_[0.0, pros[keep], 1.0]
    grid = np.linspace(0, fpr_limit, 200)
    return float(np.trapezoid(np.interp(grid, fprs, pros), grid) / fpr_limit)


def evaluate_predictions(anomaly_maps, masks, image_scores, image_labels, fpr_limit: float = 0.3) -> dict:
    pixel_scores = np.concatenate([np.asarray(item).reshape(-1) for item in anomaly_maps]).astype(np.float32)
    pixel_labels = np.concatenate([np.asarray(item).reshape(-1) for item in masks]).astype(np.uint8)
    seg_f1 = f1_max(pixel_scores, pixel_labels)
    cls_f1 = f1_max(image_scores, image_labels)
    return {
        "au_pro": pro_auc(anomaly_maps, masks, fpr_limit),
        "segmentation_au_roc": binary_auroc(pixel_scores, pixel_labels),
        "segmentation_f1_max": seg_f1["f1_max"],
        "classification_au_roc": binary_auroc(image_scores, image_labels),
        "classification_f1_max": cls_f1["f1_max"],
    }
