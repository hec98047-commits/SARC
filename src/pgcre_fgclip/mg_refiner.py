from pathlib import Path

import torch
import torch.nn as nn


def _logit(value: float):
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(value, dtype=torch.float32))


class MGPatchRefiner(nn.Module):
    """Small learnable patch-level mask predictor for MG refinement.

    FGCLIP stays frozen. This module consumes frozen dense patch features plus
    the baseline anomaly score and predicts which patches are reliable local
    anomaly candidates for the MG branch.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, initial_fusion_weight: float = 0.10):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.initial_fusion_weight = float(initial_fusion_weight)
        self.fusion_logit = nn.Parameter(_logit(initial_fusion_weight))
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, dense_feat: torch.Tensor, patch_score: torch.Tensor):
        x = build_refiner_inputs(dense_feat, patch_score)
        return self.net(x).squeeze(-1)

    def fusion_weight(self):
        return torch.sigmoid(self.fusion_logit)


def build_refiner_inputs(dense_feat: torch.Tensor, patch_score: torch.Tensor):
    patch_score = patch_score.detach().float().reshape(-1, 1)
    dense_feat = dense_feat.detach().float()
    if dense_feat.ndim != 2:
        raise RuntimeError(f"dense_feat must be [num_patches, dim], got {tuple(dense_feat.shape)}")
    if patch_score.shape[0] != dense_feat.shape[0]:
        raise RuntimeError(
            f"patch_score length mismatch: got {patch_score.shape[0]}, expected {dense_feat.shape[0]}"
        )

    score_std = patch_score.std(unbiased=False).clamp_min(1e-6)
    score_norm = (patch_score - patch_score.mean()) / score_std
    return torch.cat([dense_feat, score_norm.to(dense_feat.device)], dim=-1)


@torch.inference_mode()
def predict_patch_probs(refiner: MGPatchRefiner, dense_feat: torch.Tensor, patch_score: torch.Tensor):
    refiner.eval()
    logits = refiner(dense_feat, patch_score)
    return torch.sigmoid(logits)


def save_mg_refiner(path, refiner: MGPatchRefiner, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": refiner.state_dict(),
        "feature_dim": refiner.feature_dim,
        "hidden_dim": refiner.hidden_dim,
        "initial_fusion_weight": refiner.initial_fusion_weight,
        "fusion_weight": float(refiner.fusion_weight().detach().cpu().item()),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_mg_refiner(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    refiner = MGPatchRefiner(
        feature_dim=int(payload["feature_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        initial_fusion_weight=float(payload.get("initial_fusion_weight", 0.10)),
    )
    refiner.load_state_dict(payload["state_dict"], strict=False)
    refiner.to(device).eval()
    return refiner, payload.get("metadata", {})
