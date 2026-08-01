import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier


def normalize_map(score_map: np.ndarray):
    score_map = np.asarray(score_map, dtype=np.float32)
    mn = float(np.min(score_map))
    mx = float(np.max(score_map))
    return (score_map - mn) / (mx - mn + 1e-6)


def extract_segad_features(score_map: np.ndarray):
    """SegAD-style local statistics from an anomaly map.

    The CVPR 2024 SegAD paper uses local statistics from anomaly maps and
    segmentation maps. Here we use anomaly-map statistics only, so it plugs into
    the current FGCLIP/MG pipeline without requiring an extra segmenter.
    """

    score = normalize_map(score_map)
    features = [score]

    for size in (3, 7, 15, 31):
        mean = ndimage.uniform_filter(score, size=size, mode="reflect")
        sq_mean = ndimage.uniform_filter(score * score, size=size, mode="reflect")
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        max_v = ndimage.maximum_filter(score, size=size, mode="reflect")
        features.extend([mean, std, max_v, score - mean, max_v - mean])

    grad_x = ndimage.sobel(score, axis=1, mode="reflect")
    grad_y = ndimage.sobel(score, axis=0, mode="reflect")
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    features.append(grad)

    return np.stack(features, axis=-1).astype(np.float32, copy=False)


def sample_pixels(score_map, mask, samples_per_image, rng):
    features = extract_segad_features(score_map)
    labels = np.asarray(mask, dtype=np.uint8).reshape(-1)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_score = normalize_map(score_map).reshape(-1)

    pos_idx = np.flatnonzero(labels > 0)
    neg_idx = np.flatnonzero(labels == 0)
    if pos_idx.size == 0:
        hard_count = min(max(samples_per_image // 2, 1), neg_idx.size)
        rand_count = min(max(samples_per_image - hard_count, 1), neg_idx.size)
        hard_order = np.argsort(flat_score[neg_idx])[-hard_count:]
        chosen = np.concatenate([neg_idx[hard_order], rng.choice(neg_idx, size=rand_count, replace=False)])
    else:
        pos_count = min(pos_idx.size, max(1, samples_per_image // 2))
        neg_count = min(neg_idx.size, max(1, samples_per_image - pos_count))
        hard_count = min(neg_count, max(1, neg_count // 2))
        rand_count = max(0, neg_count - hard_count)
        hard_order = np.argsort(flat_score[neg_idx])[-hard_count:]
        chosen_neg = [neg_idx[hard_order]]
        if rand_count > 0:
            chosen_neg.append(rng.choice(neg_idx, size=rand_count, replace=False))
        chosen = np.concatenate([rng.choice(pos_idx, size=pos_count, replace=False), *chosen_neg])

    return flat_features[chosen], labels[chosen].astype(np.uint8)


def train_segad_calibrator(feature_blocks, label_blocks, max_iter=120, learning_rate=0.06, seed=7):
    x = np.concatenate(feature_blocks, axis=0)
    y = np.concatenate(label_blocks, axis=0)
    if np.unique(y).size < 2:
        raise RuntimeError("SegAD calibrator needs both normal and anomalous pixel samples.")

    clf = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=int(seed),
    )
    clf.fit(x, y)
    return clf


def predict_segad_map(score_map: np.ndarray, calibrator):
    features = extract_segad_features(score_map)
    flat = features.reshape(-1, features.shape[-1])
    probs = calibrator.predict_proba(flat)[:, 1]
    return probs.reshape(score_map.shape).astype(np.float32)


def save_segad_calibrator(path, calibrator, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"calibrator": calibrator, "metadata": metadata}, f)


def load_segad_calibrator(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["calibrator"], payload.get("metadata", {})
