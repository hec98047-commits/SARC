from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def l2_normalize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def _uniform_sample_indices(num_items: int, num_keep: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0, num_items - 1, steps=num_keep, device=device).long()


@torch.inference_mode()
def _representative_fps_indices(
    features: torch.Tensor,
    num_keep: int,
    candidate_pool_limit: int = 8192,
) -> torch.Tensor:
    num_items = int(features.shape[0])
    device = features.device
    if num_keep >= num_items:
        return torch.arange(num_items, device=device, dtype=torch.long)

    # Keep the candidate pool bounded so the side experiment stays lightweight.
    if num_items > candidate_pool_limit:
        candidate_indices = _uniform_sample_indices(num_items, candidate_pool_limit, device)
        candidate_features = features.index_select(0, candidate_indices)
    else:
        candidate_indices = torch.arange(num_items, device=device, dtype=torch.long)
        candidate_features = features

    candidate_features = l2_normalize(candidate_features.float())
    pool_size = int(candidate_features.shape[0])
    num_keep = min(int(num_keep), pool_size)

    mean_feat = l2_normalize(candidate_features.mean(dim=0, keepdim=True))[0]
    first_idx = torch.argmin(candidate_features @ mean_feat)
    selected = [int(first_idx.item())]

    min_dist = 1.0 - (candidate_features @ candidate_features[first_idx])
    for _ in range(1, num_keep):
        next_idx = torch.argmax(min_dist)
        selected_idx = int(next_idx.item())
        selected.append(selected_idx)
        new_dist = 1.0 - (candidate_features @ candidate_features[selected_idx])
        min_dist = torch.minimum(min_dist, new_dist)

    selected_local = torch.tensor(selected, device=device, dtype=torch.long)
    return candidate_indices.index_select(0, selected_local)


@torch.inference_mode()
def collect_normal_reference_patches(
    model,
    normal_image_paths,
    preprocess,
    device,
    feature_layer: int = 5,
    max_ref_patches: int = 4096,
    use_fp16: bool = True,
    resize_short_edge: int = 1024,
    max_num_patches: int = 4096,
    encode_dense_image_fn=None,
    selection_mode: str = "uniform",
) -> torch.Tensor:
    if encode_dense_image_fn is None:
        raise ValueError("encode_dense_image_fn is required for this repository's FG-CLIP models.")
    parts = []
    for image_path in normal_image_paths:
        image = Image.open(Path(image_path)).convert("RGB")
        dense_feat, _, _, _ = encode_dense_image_fn(
            image=image,
            image_processor=preprocess,
            model=model,
            resize_target=resize_short_edge,
            max_num_patches=max_num_patches,
            feature_layer=int(feature_layer),
        )
        dense_feat = l2_normalize(dense_feat.detach().float()).to(device)
        parts.append(dense_feat)
    if not parts:
        raise RuntimeError("No normal reference images were provided for NRD.")
    ref_features = torch.cat(parts, dim=0)
    max_ref_patches = int(max_ref_patches)
    if max_ref_patches > 0 and ref_features.shape[0] > max_ref_patches:
        if selection_mode == "uniform":
            index = _uniform_sample_indices(ref_features.shape[0], max_ref_patches, ref_features.device)
        elif selection_mode == "fps":
            index = _representative_fps_indices(ref_features, max_ref_patches)
        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")
        ref_features = ref_features.index_select(0, index)
    ref_features = l2_normalize(ref_features.float())
    if use_fp16 and str(device).startswith("cuda"):
        ref_features = ref_features.half()
    return ref_features


@torch.inference_mode()
def compute_normal_reference_distance(
    test_patch_features: torch.Tensor,
    ref_features: torch.Tensor,
    chunk_size: int = 8192,
    eps: float = 1e-6,
) -> torch.Tensor:
    test_patch_features = l2_normalize(test_patch_features.float(), eps=eps)
    ref_features = l2_normalize(ref_features.float(), eps=eps).to(test_patch_features.device)
    distances = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, test_patch_features.shape[0], chunk_size):
        chunk = test_patch_features[start : start + chunk_size]
        sim = chunk @ ref_features.T
        s_normal = sim.max(dim=1).values
        distances.append((1.0 - s_normal).detach().float().cpu())
    return torch.cat(distances, dim=0)


def zscore_map(x, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[finite]
    mean = float(values.mean())
    std = float(values.std())
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.isfinite(std) or std < float(eps):
        out[finite] = values - mean
    else:
        out[finite] = (values - mean) / (std + float(eps))
    return out.astype(np.float32)


def normalize_map(x, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if not finite.any():
        return out
    values = arr[finite]
    mn = float(values.min())
    mx = float(values.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < float(eps):
        out[finite] = 0.0
        return out
    out[finite] = (values - mn) / (mx - mn + float(eps))
    return out.astype(np.float32)


def sigmoid_map(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def _fuse_standard_nrd(
    H_pgcre,
    D_normal_map,
    beta: float = 0.3,
    fusion_mode: str = "add",
    eps: float = 1e-6,
) -> np.ndarray:
    h_z = zscore_map(H_pgcre, eps=eps)
    d_z = zscore_map(D_normal_map, eps=eps)
    beta = float(beta)
    if fusion_mode == "add":
        out = h_z + beta * d_z
    elif fusion_mode == "residual":
        out = h_z + beta * np.maximum(d_z, 0.0)
    elif fusion_mode == "multiply":
        out = h_z * (1.0 + beta * sigmoid_map(d_z))
    else:
        raise ValueError(f"Unsupported nrd_fusion_mode: {fusion_mode}")
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fuse_pgcre_with_nrd(
    H_pgcre,
    D_normal_map,
    beta: float = 0.3,
    fusion_mode: str = "add",
    eps: float = 1e-6,
    pro_q: float = 0.85,
    pro_beta: float = 10.0,
    eta_nrd: float = 0.10,
    lambda_prior: float = 0.03,
    normal_prior_map=None,
    return_details: bool = False,
    agree_tau: float = 0.20,
):
    if fusion_mode in {"add", "residual", "multiply"}:
        out = _fuse_standard_nrd(H_pgcre, D_normal_map, beta=beta, fusion_mode=fusion_mode, eps=eps)
        if return_details:
            details = {
                "h_pgcre_norm": normalize_map(H_pgcre, eps=eps),
                "h_nrd_fps_norm": normalize_map(out, eps=eps),
                "candidate_mask": np.zeros_like(out, dtype=np.float32),
                "normal_prior": np.zeros_like(out, dtype=np.float32),
                "foreground_map": normalize_map(out, eps=eps),
                "background_map": normalize_map(H_pgcre, eps=eps),
            }
            return out, details
        return out

    if fusion_mode == "rank_residual":
        p_norm = normalize_map(H_pgcre, eps=eps)
        h_nrd_fps = _fuse_standard_nrd(H_pgcre, D_normal_map, beta=1.0, fusion_mode="add", eps=eps)
        n_norm = normalize_map(h_nrd_fps, eps=eps)
        tau = float(np.quantile(n_norm.reshape(-1), float(pro_q)))
        gate = sigmoid_map(float(pro_beta) * (n_norm - tau))
        residual = np.maximum(n_norm - tau, 0.0).astype(np.float32)
        residual = np.square(residual).astype(np.float32)
        candidate_mask = (gate * (residual > 0.0).astype(np.float32)).astype(np.float32)
        out_raw = p_norm + float(beta) * candidate_mask * residual
        out = normalize_map(out_raw, eps=eps)
        details = {
            "h_pgcre_norm": p_norm.astype(np.float32),
            "h_nrd_fps_norm": n_norm.astype(np.float32),
            "candidate_mask": candidate_mask.astype(np.float32),
            "normal_prior": np.zeros_like(p_norm, dtype=np.float32),
            "foreground_map": out.astype(np.float32),
            "background_map": p_norm.astype(np.float32),
        }
        if return_details:
            return out, details
        return out

    if fusion_mode == "agreement_gated_add":
        p_norm = normalize_map(H_pgcre, eps=eps)
        h_nrd_fps = _fuse_standard_nrd(H_pgcre, D_normal_map, beta=beta, fusion_mode="add", eps=eps)
        n_norm = normalize_map(h_nrd_fps, eps=eps)
        p_vec = p_norm.reshape(-1).astype(np.float32)
        n_vec = n_norm.reshape(-1).astype(np.float32)
        p_vec = p_vec - float(p_vec.mean())
        n_vec = n_vec - float(n_vec.mean())
        denom = float(np.linalg.norm(p_vec) * np.linalg.norm(n_vec) + eps)
        agreement = float(np.dot(p_vec, n_vec) / denom) if denom > 0.0 else 0.0
        use_nrd = float(agreement >= float(agree_tau))
        out = normalize_map(use_nrd * n_norm + (1.0 - use_nrd) * p_norm, eps=eps)
        details = {
            "h_pgcre_norm": p_norm.astype(np.float32),
            "h_nrd_fps_norm": n_norm.astype(np.float32),
            "candidate_mask": np.full_like(p_norm, fill_value=use_nrd, dtype=np.float32),
            "normal_prior": np.zeros_like(p_norm, dtype=np.float32),
            "foreground_map": n_norm.astype(np.float32),
            "background_map": p_norm.astype(np.float32),
        }
        if return_details:
            return out, details
        return out

    if fusion_mode != "candidate_pgcre_rank":
        raise ValueError(f"Unsupported nrd_fusion_mode: {fusion_mode}")

    h_pgcre = np.asarray(H_pgcre, dtype=np.float32)
    d_normal = np.asarray(D_normal_map, dtype=np.float32)
    h_nrd_fps = _fuse_standard_nrd(h_pgcre, d_normal, beta=beta, fusion_mode="add", eps=eps)

    p_norm = normalize_map(h_pgcre, eps=eps)
    n_norm = normalize_map(h_nrd_fps, eps=eps)

    tau = float(np.quantile(n_norm.reshape(-1), float(pro_q)))
    excess = np.clip(n_norm - tau, 0.0, None).astype(np.float32)
    denom = max(1.0 - tau, float(eps))
    excess_norm = (excess / denom).astype(np.float32)

    # Keep NRD influence concentrated in a smaller high-confidence region:
    # 1) a steep gate around tau
    # 2) suppress sub-threshold activations completely
    # 3) square the normalized excess so weak above-threshold responses fade quickly
    gate = sigmoid_map(float(pro_beta) * (n_norm - tau))
    candidate_mask = (gate * (excess_norm ** 2)).astype(np.float32)
    candidate_mask = np.clip(candidate_mask, 0.0, 1.0)

    fg_map = ((1.0 - float(eta_nrd)) * p_norm + float(eta_nrd) * n_norm).astype(np.float32)

    if normal_prior_map is None:
        normal_prior = 1.0 - normalize_map(d_normal, eps=eps)
    else:
        normal_prior = normalize_map(normal_prior_map, eps=eps)
    bg_map = (p_norm - float(lambda_prior) * normal_prior).astype(np.float32)

    out_raw = candidate_mask * fg_map + (1.0 - candidate_mask) * bg_map
    out = normalize_map(out_raw, eps=eps)
    out = np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    details = {
        "h_pgcre_norm": p_norm,
        "h_nrd_fps_norm": n_norm,
        "candidate_mask": candidate_mask.astype(np.float32),
        "normal_prior": normal_prior.astype(np.float32),
        "foreground_map": fg_map.astype(np.float32),
        "background_map": bg_map.astype(np.float32),
    }
    if return_details:
        return out, details
    return out


def resize_distance_to_heatmap(distance_values, grid_size: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
    real_h, real_w = grid_size
    target_h, target_w = target_shape
    distance_map = np.asarray(distance_values, dtype=np.float32).reshape(real_h, real_w)
    return cv2.resize(distance_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
