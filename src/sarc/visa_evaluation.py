from bisect import bisect

import numpy as np
from scipy.ndimage import label


def trapezoid(x, y, x_max=None):
    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[finite_mask]
    y = y[finite_mask]

    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def compute_classification_roc(anomaly_maps, scoring_function, ground_truth_labels):
    assert len(anomaly_maps) == len(ground_truth_labels)

    anomaly_scores = list(map(scoring_function, anomaly_maps))
    num_scores = len(anomaly_maps)
    sorted_samples = sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])

    labels_np = np.asarray(ground_truth_labels)
    num_pos = labels_np[labels_np != 0].size
    num_neg = labels_np[labels_np == 0].size

    fprs = [1.0]
    tprs = [1.0]
    num_fp = num_neg
    num_tp = num_pos

    for i, (current_score, label_value) in enumerate(sorted_samples):
        if label_value == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        next_score = sorted_samples[i + 1][0] if i < num_scores - 1 else None
        if (next_score != current_score) or (next_score is None):
            fprs.append(num_fp / num_neg)
            tprs.append(num_tp / num_pos)

    return fprs[::-1], tprs[::-1]


def compute_pro(anomaly_maps, ground_truth_maps):
    print("Compute PRO curve...")

    structure = np.ones((3, 3), dtype=int)
    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps), anomaly_maps[0].shape[0], anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for component_idx in range(n_components):
            region_mask = labeled == (component_idx + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1.0 / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    anomaly_scores_flat = np.asarray(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)

    np.cumsum(fp_changes_flat, out=fp_changes_flat)
    fprs = fp_changes_flat.astype(np.float32, copy=False)
    np.divide(fprs, num_ok_pixels, out=fprs)

    np.cumsum(pro_changes_flat, out=pro_changes_flat)
    pros = pro_changes_flat
    np.divide(pros, num_gt_regions, out=pros)

    keep_mask = np.append(np.diff(anomaly_scores_flat) != 0, np.True_)
    fprs = fprs[keep_mask]
    pros = pros[keep_mask]

    np.clip(fprs, a_min=None, a_max=1.0, out=fprs)
    np.clip(pros, a_min=None, a_max=1.0, out=pros)

    zero = np.array([0.0])
    one = np.array([1.0])
    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))
