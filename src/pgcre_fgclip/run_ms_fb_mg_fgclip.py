import argparse
import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_closing, binary_dilation, label as connected_components
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "transformers")
os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT / "hub")
os.environ["HF_MODULES_CACHE"] = str(CACHE_ROOT / "modules")

from fgclip_ad.datasets import discover_classes, load_dataset_samples, load_mask  # noqa: E402
from fgclip_ad.metrics import f1_max, pro_auc  # noqa: E402
from fgclip_ad.psme_ms_modules import (  # noqa: E402
    apply_text_conditioned_psme,
    compute_text_conditioned_mask,
    compute_text_conditioned_score,
    ensure_finite_array,
    global_guided_multiscale_fusion,
    patch_score_to_heatmap,
    robust_normalize_heatmap,
    save_intermediate_heatmap,
)
from fgclip_ad.utils import make_relative_safe_name  # noqa: E402
from hpmr.ncrs_selection import (  # noqa: E402
    calibrate_risk_threshold,
    select_response as ncrs_select_response,
)
from hpmr.response_calibration import calibrate_response  # noqa: E402
from hpmr.reliability_fusion import (  # noqa: E402
    compute_reliability_weight,
    reliability_aware_positive_fusion,
)
from hpmr.safe_response_selection import safe_select_response  # noqa: E402
from run_mvtec_ad_benchmark import (  # noqa: E402
    GOOD_BANK_CHUNK_SIZE,
    GOOD_BANK_MAX_PATCHES,
    GOOD_BANK_PATCHES_PER_IMAGE,
    MAX_NUM_PATCHES,
    MIN_THRESHOLD_MARGIN,
    PRO_INTEGRATION_LIMIT,
    RESIZE_SHORT_EDGE,
    THRESHOLD_STD_MULT,
    activate_object_mg_refiner,
    attach_mg_refiner,
    build_good_prototype,
    calibrate_threshold,
    compute_binary_auroc,
    compute_maps_and_score,
    encode_dense_image,
    encode_text,
    get_prompts as get_mvtec_prompts,
    iter_progress,
    load_model,
    set_progress_style,
    standardize_map,
)
from run_visa_benchmark import get_prompts as get_visa_prompts  # noqa: E402
from visa_evaluation import compute_pro as legacy_compute_pro, trapezoid as legacy_trapezoid  # noqa: E402


BASELINE_METHOD = "baseline"
MG_METHOD_NO_REFINER = "mg_positive_w0.06_ratio0.05_d1_l3"
MG_METHOD_REFINER = "mg_positive_w0.06_ratio0.05_d1_l3_refiner"
MS_METHOD = "ms_fb_mg_fgclip"
HPMR_METHODS = ("fgclip_baseline", "prompt", "prompt_mask", "prompt_mask_ms", "full")
LEGACY_METHODS = (
    "baseline",
    "fgclip_baseline",
    MG_METHOD_NO_REFINER,
    MG_METHOD_REFINER,
    "mg_positive",
    "mg_only",
    "mg_fg",
    "mg_ms",
    "mg_fg_ms",
    MS_METHOD,
    "mg_ms_hybrid_prompt",
    "full_ours",
)
ALL_METHODS = HPMR_METHODS + LEGACY_METHODS
DEFAULT_METRIC_BINS = 4096
DEFAULT_METRIC_EXACT_MAX_PIXELS = 70_000_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MS-FB-MG-FGCLIP inference enhancement.",
        epilog=(
            "Example (do not run on private data unless you explicitly provide the paths):\n"
            "python run_ms_fb_mg_fgclip.py ^\n"
            "  --dataset visa ^\n"
            "  --data_root D:/your_data_root ^\n"
            "  --model_path D:/models/FGCLIP ^\n"
            "  --output_dir outputs/psme_gms_fgclip ^\n"
            "  --text_conditioned_mask ^\n"
            "  --global_guided_fusion ^\n"
            "  --tile_weight_temperature 1.0 ^\n"
            "  --tile_topk_ratio 0.10 ^\n"
            "  --lambda_weight 0.10 ^\n"
            "  --alpha 0.50 ^\n"
            "  --save_intermediate"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "models" / "FGCLIP")
    parser.add_argument("--mg_model_path", type=Path, default=REPO_ROOT / "models" / "MGFGGCLIP")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "ms_fb_mg_fgclip")
    parser.add_argument("--method", choices=ALL_METHODS, default=MS_METHOD)
    parser.add_argument("--method_name", default=None, help="Optional explicit output method name.")
    parser.add_argument("--classes", "--objects", nargs="+", default=None)
    parser.add_argument("--visa_split", type=Path, default=None)
    parser.add_argument("--tile_mode", default="2x2", choices=["2x2", "3x3", "4x4"])
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--lambda_weight", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--q", type=float, default=0.80)
    parser.add_argument("--foreground_mode", default="heatmap_topk", choices=["none", "heatmap_topk", "image_foreground"])
    parser.add_argument(
        "--foreground_heatmap_quantile",
        type=float,
        default=0.88,
        help="High-response quantile used to tighten image_foreground with heatmap support.",
    )
    parser.add_argument(
        "--foreground_min_area",
        type=float,
        default=0.005,
        help="Minimum area ratio for the refined foreground candidate mask.",
    )
    parser.add_argument(
        "--foreground_max_area",
        type=float,
        default=0.45,
        help="Maximum area ratio for the refined foreground candidate mask before fallback tightening.",
    )
    parser.add_argument("--text_conditioned_mask", dest="text_conditioned_mask", action="store_true")
    parser.add_argument("--enable_text_conditioned_mask", dest="text_conditioned_mask", action="store_true")
    parser.add_argument("--disable_text_conditioned_mask", dest="text_conditioned_mask", action="store_false")
    parser.add_argument("--global_guided_fusion", action="store_true")
    parser.add_argument("--tile_weight_temperature", type=float, default=1.0)
    parser.add_argument("--tile_topk_ratio", type=float, default=0.10)
    parser.add_argument("--text_mask_tau", type=float, default=0.5)
    parser.add_argument("--text_mask_beta", type=float, default=0.15)
    parser.add_argument("--response_calibration", action="store_true")
    parser.add_argument("--response_calibration_alpha", type=float, default=0.15)
    parser.add_argument("--response_calibration_positive_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reliability_ms_fusion", action="store_true")
    parser.add_argument("--reliability_beta", type=float, default=0.08)
    parser.add_argument("--reliability_topk_ratio", type=float, default=0.05)
    parser.add_argument("--reliability_min_weight", type=float, default=0.0)
    parser.add_argument("--reliability_max_weight", type=float, default=1.0)
    parser.add_argument("--safe_response_selection", action="store_true")
    parser.add_argument("--safe_selection_margin", type=float, default=0.02)
    parser.add_argument("--safe_selection_topk_ratio", type=float, default=0.05)
    parser.add_argument("--normal_calibrated_selection", action="store_true")
    parser.add_argument("--pars_risk_quantile", type=float, default=0.95)
    parser.add_argument("--pars_margin", type=float, default=0.0)
    parser.add_argument("--pars_topk_ratio", type=float, default=0.05)
    parser.add_argument("--pars_area_q", type=float, default=0.80)
    parser.add_argument("--pars_save_selection_stats", action="store_true")
    parser.add_argument("--pars_enhanced_bias", type=float, default=0.0)
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--save_selected_visualizations_only", action="store_true")
    parser.add_argument(
        "--visualization_only",
        action="store_true",
        help="Only export selected visualizations and skip metric aggregation. When enough samples are saved for a class, stop that class early.",
    )
    parser.add_argument(
        "--save_selected_include_good",
        action="store_true",
        help="Also save selected_visualizations for normal/good test samples.",
    )
    parser.add_argument("--vis_classes", default=None, help="Comma-separated class names for selected qualitative export.")
    parser.add_argument("--max_vis_per_class", type=int, default=0, help="Maximum saved selected test samples per class.")
    parser.add_argument("--save_npy_visualizations", action="store_true", help="Also save float32 .npy heatmaps for selected qualitative export.")
    parser.add_argument("--vis_overlay_alpha", type=float, default=0.38, help="Overlay alpha used for selected qualitative visualizations.")
    parser.add_argument("--vis_percentile_low", type=float, default=70.0, help="Lower percentile used to stretch visualization heatmaps.")
    parser.add_argument("--vis_percentile_high", type=float, default=99.2, help="Upper percentile used to stretch visualization heatmaps.")
    parser.add_argument("--vis_gamma", type=float, default=0.85, help="Gamma used after percentile stretching for visualization heatmaps.")
    parser.add_argument(
        "--vis_activation_percentile",
        type=float,
        default=88.0,
        help="Only responses above this percentile are emphasized in qualitative heatmaps and overlays.",
    )
    parser.add_argument("--save_heatmaps", action="store_true", help="Save per-method raw heatmaps as float32 TIFF files.")
    parser.add_argument("--max_test_images", type=int, default=None)
    parser.add_argument("--subset", choices=["all", "small", "tiny"], default="all")
    parser.add_argument("--anomaly_only_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--eval_subsets",
        default=None,
        help="Comma-separated subsets to evaluate after one inference pass, e.g. all,small,tiny.",
    )
    parser.add_argument(
        "--eval_all_subsets",
        action="store_true",
        help="Evaluate all/small/tiny after one inference pass without repeating inference.",
    )
    parser.add_argument("--resize_short_edge", type=int, default=RESIZE_SHORT_EDGE)
    parser.add_argument("--max_num_patches", type=int, default=MAX_NUM_PATCHES)
    parser.add_argument("--pro_limit", type=float, default=PRO_INTEGRATION_LIMIT)
    parser.add_argument("--metric_bins", type=int, default=DEFAULT_METRIC_BINS)
    parser.add_argument("--metric_exact_max_pixels", type=int, default=DEFAULT_METRIC_EXACT_MAX_PIXELS)
    parser.add_argument("--progress_style", choices=["stage", "live"], default="live")
    parser.add_argument("--mg_refiner_checkpoint", type=Path, default=None)
    parser.add_argument("--mg_refiner_dir", type=Path, default=None)
    parser.add_argument("--mg_fusion_weight", type=float, default=0.10)
    parser.add_argument("--mg_fusion_mode", choices=["positive", "direct", "off"], default="positive")
    parser.add_argument(
        "--mg_mask_type",
        choices=["hard", "soft", "hybrid"],
        default="hard",
        help="Optional script-level MG fusion mask type. Default hard preserves the previous behavior.",
    )
    parser.add_argument(
        "--soft_mask_gamma",
        type=float,
        default=0.5,
        help="Hard-mask weight used when --mg_mask_type hybrid.",
    )
    parser.add_argument(
        "--candidate_mask_source",
        choices=["baseline", "tile", "text"],
        default="baseline",
        help="Heatmap source used to derive the candidate mask.",
    )
    parser.add_argument(
        "--candidate_mask_mode",
        choices=["quantile", "topk_regions"],
        default="quantile",
        help="How the candidate mask is derived from the selected source heatmap.",
    )
    parser.add_argument(
        "--candidate_topk_ratio",
        type=float,
        default=0.03,
        help="Top-k ratio used by the topk_regions candidate mask mode.",
    )
    parser.add_argument(
        "--candidate_max_regions",
        type=int,
        default=5,
        help="Maximum connected regions retained by the topk_regions candidate mask mode.",
    )
    parser.add_argument(
        "--iterative_refinement_steps",
        type=int,
        default=1,
        help="Number of candidate-mask refinement iterations for MS fusion. Use 1 to disable iterative refinement.",
    )
    parser.add_argument(
        "--iterative_refinement_tol",
        type=float,
        default=1e-3,
        help="Stop iterative refinement early when the mean absolute map change falls below this value.",
    )
    parser.add_argument("--mg_mask_ratio", type=float, default=0.05)
    parser.add_argument("--mg_start_layer", type=int, default=3)
    parser.add_argument("--mg_end_layer", type=int, default=None)
    parser.add_argument("--mg_mask_threshold", type=float, default=0.5)
    parser.add_argument("--mg_neg_bias", type=float, default=-1e4)
    parser.add_argument(
        "--attention_bias_eta",
        type=float,
        default=1.0,
        help="Strength multiplier applied to the additive mask-guided attention bias.",
    )
    parser.add_argument("--dual_layer_guidance", action="store_true")
    parser.add_argument("--local_feature_layer", type=int, default=5)
    parser.add_argument("--semantic_mask_weight", type=float, default=0.7)
    parser.add_argument("--local_mask_weight", type=float, default=0.3)
    parser.add_argument(
        "--local_mask_source",
        choices=["fused", "local", "intersect"],
        default="fused",
        help="Source used to build the semantic-local mask seed for candidate guidance.",
    )
    parser.add_argument("--save_dual_layer_debug", action="store_true")
    parser.add_argument("--dual_layer_debug_dir", type=Path, default=None)
    parser.add_argument(
        "--prompt_ensemble",
        action="store_true",
        help="Use multi-granularity prompt ensemble and average prompt features into one normalized text feature.",
    )
    parser.add_argument(
        "--prompt_style",
        choices=["default", "spatial_aware", "adaptive", "pcb_specific", "mvtec_sota"],
        default="default",
        help="Prompt construction style used when prompt ensemble is enabled.",
    )
    parser.add_argument(
        "--adaptive_prompt_policy_path",
        type=Path,
        default=None,
        help="Optional JSON policy for category-adaptive prompt selection when --prompt_style adaptive.",
    )
    parser.add_argument(
        "--aupro_segauc_only",
        action="store_true",
        help="Only compute AU-PRO and P-AUROC/SegAUC; skip SegF1, PixelAP, and image-level metrics.",
    )
    parser.add_argument("--enable_mg", dest="enable_mg", action="store_true")
    parser.add_argument("--disable_mg", dest="enable_mg", action="store_false")
    parser.add_argument("--enable_fg", dest="enable_fg", action="store_true")
    parser.add_argument("--disable_fg", dest="enable_fg", action="store_false")
    parser.add_argument("--enable_ms", dest="enable_ms", action="store_true")
    parser.add_argument("--disable_ms", dest="enable_ms", action="store_false")
    parser.add_argument("--enable_positive_fusion", dest="enable_positive_fusion", action="store_true")
    parser.add_argument("--disable_positive_fusion", dest="enable_positive_fusion", action="store_false")
    parser.add_argument("--dry_run", action="store_true", help="Resolve and print method flags without loading data or models.")
    parser.set_defaults(
        enable_mg=None,
        enable_fg=None,
        enable_ms=None,
        enable_positive_fusion=None,
        text_conditioned_mask=False,
    )
    args = parser.parse_args()
    argv = sys.argv[1:]
    args._user_requested_fg = "--enable_fg" in argv
    args._user_disabled_fg = "--disable_fg" in argv
    args._user_requested_foreground_mode = any(
        item == "--foreground_mode" or item.startswith("--foreground_mode=") for item in argv
    )
    return args


def resolve_mg_refiner_runtime(args):
    refiner_dir = args.mg_refiner_dir
    if refiner_dir is None:
        args.mg_refiner_checkpoint = None
        return None, MG_METHOD_NO_REFINER

    refiner_dir = refiner_dir.resolve()
    if not refiner_dir.exists():
        print(f"[WARN] mg_refiner_dir not found, using no-refiner MG branch: {refiner_dir}")
        args.mg_refiner_checkpoint = None
        return None, MG_METHOD_NO_REFINER

    return refiner_dir, MG_METHOD_REFINER


def normalize_map(x):
    array = np.asarray(x, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(array))
    mx = float(np.max(array))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - mn) / (mx - mn)).astype(np.float32)


def resize_map(anomaly_map, size):
    return np.asarray(Image.fromarray(np.asarray(anomaly_map, dtype=np.float32), mode="F").resize(size, Image.BICUBIC))


def minmax_normalize_map(x):
    array = np.asarray(x, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(array))
    mx = float(np.max(array))
    denom = max(mx - mn, 1e-8)
    return ((array - mn) / denom).astype(np.float32)


def build_mg_mask_from_heatmap(heatmap, q=0.8, mask_type="hard", gamma=0.5):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim < 2:
        raise ValueError(f"heatmap must be at least 2D, got shape={heatmap.shape}")
    q = min(max(float(q), 0.0), 1.0)
    gamma = min(max(float(gamma), 0.0), 1.0)
    h_norm = minmax_normalize_map(heatmap)
    threshold = float(np.quantile(h_norm, q))
    m_hard = (h_norm >= threshold).astype(np.float32)
    m_soft = h_norm.astype(np.float32)
    if mask_type == "hard":
        mask = m_hard
    elif mask_type == "soft":
        mask = m_soft
    elif mask_type == "hybrid":
        mask = gamma * m_hard + (1.0 - gamma) * m_soft
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask, threshold


def build_topk_region_mask_from_heatmap(heatmap, topk_ratio=0.03, max_regions=5, mask_type="hard", gamma=0.5):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim < 2:
        raise ValueError(f"heatmap must be at least 2D, got shape={heatmap.shape}")
    topk_ratio = min(max(float(topk_ratio), 1e-4), 0.5)
    max_regions = max(int(max_regions), 1)
    h_norm = minmax_normalize_map(heatmap)
    threshold = float(np.quantile(h_norm, max(0.0, 1.0 - topk_ratio)))
    binary_mask = h_norm >= threshold
    binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3), dtype=bool))
    labeled, num = connected_components(binary_mask.astype(np.uint8))
    if num <= 0:
        return build_mg_mask_from_heatmap(heatmap, q=max(0.0, 1.0 - topk_ratio), mask_type=mask_type, gamma=gamma)
    region_areas = []
    for label_id in range(1, num + 1):
        area = int(np.sum(labeled == label_id))
        if area > 0:
            region_areas.append((area, label_id))
    if not region_areas:
        return build_mg_mask_from_heatmap(heatmap, q=max(0.0, 1.0 - topk_ratio), mask_type=mask_type, gamma=gamma)
    region_areas.sort(reverse=True)
    kept_mask = np.zeros_like(h_norm, dtype=bool)
    for _, label_id in region_areas[:max_regions]:
        kept_mask |= labeled == label_id
    kept_mask = binary_dilation(kept_mask, structure=np.ones((3, 3), dtype=bool))
    m_hard = kept_mask.astype(np.float32)
    m_soft = (h_norm * m_hard).astype(np.float32)
    gamma = min(max(float(gamma), 0.0), 1.0)
    if mask_type == "hard":
        mask = m_hard
    elif mask_type == "soft":
        mask = m_soft
    elif mask_type == "hybrid":
        mask = gamma * m_hard + (1.0 - gamma) * m_soft
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")
    return np.clip(mask, 0.0, 1.0).astype(np.float32), threshold


def select_candidate_mask_source_heatmap(
    args,
    default_heatmap,
    tile_heatmap=None,
    text_heatmap=None,
):
    source = str(getattr(args, "candidate_mask_source", "baseline")).strip().lower()
    if source == "tile":
        if tile_heatmap is not None:
            return np.asarray(tile_heatmap, dtype=np.float32), "tile"
        return np.asarray(default_heatmap, dtype=np.float32), "baseline_fallback"
    if source == "text":
        if text_heatmap is not None:
            return np.asarray(text_heatmap, dtype=np.float32), "text"
        return np.asarray(default_heatmap, dtype=np.float32), "baseline_fallback"
    return np.asarray(default_heatmap, dtype=np.float32), "baseline"


def build_candidate_mask_from_heatmap(heatmap, args):
    mode = str(getattr(args, "candidate_mask_mode", "quantile")).strip().lower()
    if mode == "topk_regions":
        return build_topk_region_mask_from_heatmap(
            heatmap,
            topk_ratio=getattr(args, "candidate_topk_ratio", 0.03),
            max_regions=getattr(args, "candidate_max_regions", 5),
            mask_type=getattr(args, "mg_mask_type", "hard"),
            gamma=getattr(args, "soft_mask_gamma", 0.5),
        )
    return build_mg_mask_from_heatmap(
        heatmap,
        q=getattr(args, "q", 0.8),
        mask_type=getattr(args, "mg_mask_type", "hard"),
        gamma=getattr(args, "soft_mask_gamma", 0.5),
    )


def parse_tile_mode(mode: str) -> int:
    parts = str(mode).lower().split("x")
    if len(parts) != 2 or parts[0] != parts[1]:
        raise ValueError(f"Unsupported tile mode: {mode}")
    try:
        grid_size = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"Unsupported tile mode: {mode}") from exc
    if grid_size < 1:
        raise ValueError(f"Unsupported tile mode: {mode}")
    return grid_size


def make_overlapping_tiles(image, mode="2x2", overlap=0.25):
    grid_size = parse_tile_mode(mode)
    width, height = image.size
    max_overlap = max(0.0, (grid_size - 1) / max(grid_size, 1) - 1e-6)
    overlap = min(max(float(overlap), 0.0), max_overlap)
    denom = max(1e-6, grid_size - (grid_size - 1) * overlap)
    tile_w = int(math.ceil(width / denom))
    tile_h = int(math.ceil(height / denom))
    if grid_size == 1:
        x_positions = [0]
        y_positions = [0]
    else:
        x_step = (width - tile_w) / float(grid_size - 1)
        y_step = (height - tile_h) / float(grid_size - 1)
        x_positions = [int(round(i * x_step)) for i in range(grid_size)]
        y_positions = [int(round(i * y_step)) for i in range(grid_size)]
    tiles = []
    seen = set()
    for y1 in y_positions:
        for x1 in x_positions:
            x2 = min(width, x1 + tile_w)
            y2 = min(height, y1 + tile_h)
            x1 = max(0, x2 - tile_w)
            y1 = max(0, y2 - tile_h)
            key = (x1, y1, x2, y2)
            if key in seen:
                continue
            seen.add(key)
            tiles.append(
                {
                    "image": image.crop((x1, y1, x2, y2)),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "original_size": (width, height),
                }
            )
    return tiles


def paste_tile_heatmaps(tile_heatmaps, image_h, image_w):
    canvas = np.full((image_h, image_w), -np.inf, dtype=np.float32)
    covered = np.zeros((image_h, image_w), dtype=bool)
    for item in tile_heatmaps:
        heatmap = item["heatmap"]
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        resized = resize_map(heatmap, (x2 - x1, y2 - y1)).astype(np.float32)
        current = canvas[y1:y2, x1:x2]
        current_covered = covered[y1:y2, x1:x2]
        canvas[y1:y2, x1:x2] = np.where(current_covered, np.maximum(current, resized), resized)
        covered[y1:y2, x1:x2] = True
    canvas[~covered] = 0.0
    return canvas


def otsu_threshold(gray):
    values = np.asarray(gray, dtype=np.float32).reshape(-1)
    if float(values.max() - values.min()) < 1e-6:
        return None
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    prob = hist.astype(np.float64) / max(int(hist.sum()), 1)
    omega = np.cumsum(prob)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma = np.divide((mu_t * omega - mu) ** 2, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(centers[int(np.argmax(sigma))])


def make_foreground_mask(
    image,
    A_full,
    mode="heatmap_topk",
    heatmap_quantile=0.88,
    min_area=0.005,
    max_area=0.45,
):
    height, width = A_full.shape
    if mode == "none":
        return np.ones((height, width), dtype=np.float32)

    def _largest_component(binary_mask):
        labeled, num = connected_components(binary_mask.astype(np.uint8))
        if num <= 1:
            return binary_mask
        areas = []
        for label_id in range(1, num + 1):
            area = int(np.sum(labeled == label_id))
            areas.append((area, label_id))
        _, best_id = max(areas, key=lambda item: item[0])
        return (labeled == best_id)

    def _finalize(binary_mask, min_area_ratio, max_area_ratio):
        binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3), dtype=bool))
        binary_mask = _largest_component(binary_mask)
        area_ratio = float(binary_mask.mean())
        if area_ratio < min_area_ratio:
            return None
        if area_ratio > max_area_ratio:
            return None
        return binary_mask.astype(np.float32)

    if mode == "heatmap_topk":
        full = normalize_map(A_full)
        threshold = float(np.quantile(full, 0.70))
        mask = full >= threshold
    elif mode == "image_foreground":
        gray = np.asarray(ImageOps.grayscale(image).resize((width, height), Image.BICUBIC), dtype=np.float32) / 255.0
        threshold = otsu_threshold(gray)
        if threshold is None:
            return np.ones((height, width), dtype=np.float32)
        bright = gray >= threshold
        dark = gray <= threshold
        image_mask = bright if bright.mean() <= 0.85 else dark
        if image_mask.mean() < 0.02 or image_mask.mean() > 0.98:
            return np.ones((height, width), dtype=np.float32)
        full = normalize_map(A_full)
        high_q = float(np.quantile(full, min(max(float(heatmap_quantile), 0.50), 0.995)))
        tighter_q = float(np.quantile(full, min(max(float(heatmap_quantile) + 0.05, 0.55), 0.995)))
        heat_core = full >= high_q
        heat_tight = full >= tighter_q
        mask = image_mask & heat_core
        refined = _finalize(mask, min_area, max_area)
        if refined is None:
            refined = _finalize(image_mask & heat_tight, min_area, max_area)
        if refined is None:
            refined = _finalize(heat_core, min_area, max_area)
        if refined is None:
            refined = _finalize(heat_tight, min_area, max_area)
        if refined is None:
            return np.ones((height, width), dtype=np.float32)
        return refined
    else:
        raise ValueError(f"Unsupported foreground mode: {mode}")

    mask = binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = binary_dilation(mask, structure=np.ones((7, 7), dtype=bool))
    if mask.mean() < 0.005:
        return np.ones((height, width), dtype=np.float32)
    return mask.astype(np.float32)


def high_confidence_fusion(A_full, A_tile, M_fg, lambda_weight=0.5, q=0.8):
    full = np.asarray(A_full, dtype=np.float32)
    tile = np.asarray(A_tile, dtype=np.float32)
    threshold = float(np.quantile(tile, min(max(float(q), 0.0), 1.0)))
    A_boost = tile.astype(np.float32, copy=True)
    A_boost -= threshold
    np.maximum(A_boost, 0.0, out=A_boost)
    A_boost *= M_fg.astype(np.float32, copy=False)
    A_boost *= float(lambda_weight)
    result = full.astype(np.float32, copy=True)
    result += A_boost
    return result


def direct_fusion(base_map, candidate_map, lambda_weight=0.5, mask=None):
    base = np.asarray(base_map, dtype=np.float32)
    candidate = np.asarray(candidate_map, dtype=np.float32)
    result = base + float(lambda_weight) * (candidate - base)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        result = base + float(lambda_weight) * mask * (candidate - base)
    return result.astype(np.float32)


def positive_residual_fusion(base_map, candidate_map, lambda_weight=0.5, mask=None, q=None):
    base = np.asarray(base_map, dtype=np.float32)
    candidate = np.asarray(candidate_map, dtype=np.float32)
    if q is not None:
        q = min(max(float(q), 0.0), 1.0)
        threshold = float(np.quantile(candidate, q))
        candidate = np.where(candidate >= threshold, candidate, base)
    delta = np.maximum(candidate - base, 0.0)
    if mask is not None:
        delta *= np.asarray(mask, dtype=np.float32)
    return (base + float(lambda_weight) * delta).astype(np.float32)


def compose_fusion_mask(foreground_mask, candidate_mask, mg_mask_type):
    fusion_mask = foreground_mask
    if mg_mask_type != "hard":
        fusion_mask = candidate_mask if fusion_mask is None else (np.asarray(fusion_mask, dtype=np.float32) * np.asarray(candidate_mask, dtype=np.float32)).astype(np.float32)
    return fusion_mask


def iterative_refine_multiscale_fusion(
    base_map,
    candidate_map,
    args,
    foreground_mask=None,
    initial_candidate_mask=None,
):
    max_steps = max(int(getattr(args, "iterative_refinement_steps", 1)), 1)
    trace = []
    current_map = None
    prev_map = None
    candidate_mask = None if initial_candidate_mask is None else np.asarray(initial_candidate_mask, dtype=np.float32)
    current_source_heatmap = np.asarray(base_map, dtype=np.float32)
    last_threshold = None
    for step_idx in range(max_steps):
        if step_idx > 0:
            candidate_mask, last_threshold = build_candidate_mask_from_heatmap(current_source_heatmap, args)
        fusion_mask = compose_fusion_mask(foreground_mask, candidate_mask, getattr(args, "mg_mask_type", "hard"))
        if getattr(args, "enable_positive_fusion", False):
            current_map = positive_residual_fusion(
                base_map,
                candidate_map,
                lambda_weight=args.lambda_weight,
                mask=fusion_mask,
                q=args.q,
            )
        else:
            current_map = direct_fusion(
                base_map,
                candidate_map,
                lambda_weight=args.lambda_weight,
                mask=fusion_mask,
            )
        delta = None
        if prev_map is not None:
            delta = float(np.mean(np.abs(current_map - prev_map)))
        trace.append(
            {
                "step": int(step_idx + 1),
                "delta": delta,
                "threshold": None if last_threshold is None else float(last_threshold),
            }
        )
        if delta is not None and delta < float(getattr(args, "iterative_refinement_tol", 1e-3)):
            break
        prev_map = np.asarray(current_map, dtype=np.float32)
        current_source_heatmap = prev_map
    return current_map, candidate_mask, trace


def dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(item))
    return out


MVTec_PROMPTAD_STYLE_DEFECTS = {
    "bottle": ["large breakage", "small breakage", "contamination", "crack"],
    "cable": ["bent wire", "missing part", "missing wire", "cut", "poke defect"],
    "capsule": ["crack", "faulty imprint", "poke defect", "scratch", "compression deformation"],
    "carpet": ["hole", "color stain", "metal contamination", "thread residue", "cut"],
    "grid": ["breakage", "thread residue", "metal contamination", "glue residue", "bent shape"],
    "hazelnut": ["crack", "cut", "hole", "abnormal print"],
    "leather": ["color stain", "cut", "fold", "glue residue", "poke defect"],
    "metal_nut": ["bent shape", "color stain", "flipped orientation", "scratch"],
    "pill": ["color stain", "contamination", "crack", "faulty imprint", "scratch", "abnormal type"],
    "screw": ["manipulated front", "scratch neck", "scratch head", "local scratch"],
    "tile": ["crack", "glue strip", "gray stroke", "oil stain", "rough surface"],
    "toothbrush": ["abnormal bristle pattern", "damaged bristles", "local defect"],
    "transistor": ["bent lead", "cut lead", "misplaced lead", "surface damage"],
    "wood": ["color stain", "hole", "scratch", "liquid stain"],
    "zipper": ["broken teeth", "fabric border defect", "broken fabric", "split teeth", "squeezed teeth"],
}

ANOMALYCLIP_TEMPLATE_CORE = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a cropped photo of a {}",
    "a good photo of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a blurry photo of a {}",
    "a low resolution photo of a {}",
    "a jpeg corrupted photo of a {}",
    "there is a {} in the scene",
    "this is a {} in the scene",
]

ANOMALYCLIP_NORMAL_STATES = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

ANOMALYCLIP_ABNORMAL_STATES = [
    "damaged {}",
    "broken {}",
    "abnormal {}",
    "blemished {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

MVTec_TARGETED_PROMPTS = {
    "bottle": {
        "normal": [
            "a normal bottle with an intact rim and smooth glass surface",
            "a clean bottle without crack, chip, or contamination",
            "a bottle with complete shape and uniform appearance",
        ],
        "abnormal": [
            "a bottle with a small crack on the glass surface",
            "a bottle with large breakage on the rim or body",
            "a bottle with contamination or foreign residue",
            "a bottle with a local chipped region",
        ],
    },
    "cable": {
        "normal": [
            "a normal cable with continuous insulation and aligned wires",
            "an intact cable with complete wire structure and clean surface",
            "a cable without cut, break, missing wire, or puncture",
        ],
        "abnormal": [
            "a cable with a bent wire near the local region",
            "a cable with missing wire or incomplete strand structure",
            "a cable with a small cut on the insulation surface",
            "a cable with a local poke hole or puncture defect",
            "a cable with exposed inner wire in a small area",
        ],
    },
    "capsule": {
        "normal": [
            "a normal capsule with smooth coating and clear imprint",
            "an intact capsule without crack, scratch, or compression",
            "a capsule with uniform color and clean surface",
        ],
        "abnormal": [
            "a capsule with a local crack on the shell",
            "a capsule with a faulty or abnormal imprint",
            "a capsule with a small scratch on the surface",
            "a capsule with local compression or squeezed shape",
            "a capsule with a tiny poke defect",
        ],
    },
    "carpet": {
        "normal": [
            "a normal carpet with uniform texture and continuous woven pattern",
            "a clean carpet surface without stain, hole, or loose thread",
            "a carpet with regular texture and no local damage",
        ],
        "abnormal": [
            "a carpet with a local hole in the texture",
            "a carpet with color stain on a small region",
            "a carpet with metal contamination or foreign residue",
            "a carpet with loose thread or thread residue",
            "a carpet with a local cut in the surface pattern",
        ],
    },
    "grid": {
        "normal": [
            "a normal metal grid with regular pattern and intact cells",
            "an intact grid with uniform structure and clean surface",
            "a grid without bent shape, breakage, or residue",
        ],
        "abnormal": [
            "a grid with local breakage in the mesh pattern",
            "a grid with bent shape on a small region",
            "a grid with glue residue or metal contamination",
            "a grid with thread residue attached to the surface",
        ],
    },
    "hazelnut": {
        "normal": [
            "a normal hazelnut with intact shell and natural texture",
            "a hazelnut without cut, hole, crack, or abnormal print",
            "a clean hazelnut surface with uniform appearance",
        ],
        "abnormal": [
            "a hazelnut with a local crack on the shell",
            "a hazelnut with a small cut or hole",
            "a hazelnut with abnormal printed mark on the surface",
        ],
    },
    "leather": {
        "normal": [
            "normal leather with uniform grain and clean surface",
            "leather without cut, fold, glue residue, or stain",
            "a leather surface with smooth continuous texture",
        ],
        "abnormal": [
            "leather with a local cut on the surface",
            "leather with a fold or crease defect",
            "leather with glue residue or color stain",
            "leather with a tiny poke defect",
        ],
    },
    "metal_nut": {
        "normal": [
            "a normal metal nut with regular hexagonal contour and centered hole",
            "an intact metal nut with complete thread and uniform metallic surface",
            "a metal nut without deformation, scratch, stain, or flipped orientation",
        ],
        "abnormal": [
            "a metal nut with a bent or deformed contour",
            "a metal nut with flipped orientation in the local view",
            "a metal nut with a local scratch on the rim or surface",
            "a metal nut with color stain on the metal surface",
            "a metal nut with damaged thread near the hole",
        ],
    },
    "pill": {
        "normal": [
            "a normal pill with smooth coating and regular imprint",
            "an intact pill without crack, stain, contamination, or scratch",
            "a pill with uniform color and complete shape",
        ],
        "abnormal": [
            "a pill with a local crack on the body",
            "a pill with abnormal or faulty imprint",
            "a pill with color stain or contamination on the surface",
            "a pill with a small scratch or chipped local region",
            "a pill with abnormal type or mismatched appearance",
        ],
    },
    "screw": {
        "normal": [
            "a normal screw with intact head, regular thread, and clean neck",
            "an intact screw without scratch, dent, or damaged front",
            "a screw with complete metallic structure and no local defect",
        ],
        "abnormal": [
            "a screw with local scratch on the head",
            "a screw with local scratch on the neck",
            "a screw with damaged or manipulated front region",
            "a screw with a small local dent on the surface",
        ],
    },
    "tile": {
        "normal": [
            "a normal tile with uniform surface and continuous texture",
            "a clean tile without crack, stain, glue strip, or rough region",
            "a tile with smooth appearance and no local defect",
        ],
        "abnormal": [
            "a tile with a local crack on the surface",
            "a tile with oil stain or gray stroke on a small area",
            "a tile with glue strip residue",
            "a tile with rough surface in a local patch",
        ],
    },
    "toothbrush": {
        "normal": [
            "a normal toothbrush with aligned bristles and intact head",
            "a clean toothbrush without abnormal bristle pattern or local damage",
            "a toothbrush with uniform bristle arrangement",
        ],
        "abnormal": [
            "a toothbrush with abnormal bristle pattern in a local region",
            "a toothbrush with damaged or missing bristles",
            "a toothbrush with a small local defect on the head",
        ],
    },
    "transistor": {
        "normal": [
            "a normal transistor with straight aligned leads and intact package",
            "an intact transistor with complete metal pins and clean body surface",
            "a transistor without bent lead, cut lead, or misplaced pin",
        ],
        "abnormal": [
            "a transistor with a bent lead in the local region",
            "a transistor with a cut or missing lead",
            "a transistor with misplaced or misaligned lead position",
            "a transistor with local damage on the package surface",
            "a transistor with incomplete pin structure",
        ],
    },
    "wood": {
        "normal": [
            "normal wood with uniform grain and clean natural texture",
            "wood without stain, hole, scratch, or liquid mark",
            "a wood surface with continuous texture and no local defect",
        ],
        "abnormal": [
            "wood with local color stain on the surface",
            "wood with a small hole in the texture",
            "wood with a local scratch mark",
            "wood with liquid stain in a small area",
        ],
    },
    "zipper": {
        "normal": [
            "a normal zipper with aligned teeth and intact fabric border",
            "an intact zipper without missing tooth, torn fabric, or split teeth",
            "a zipper with continuous tooth pattern and clean cloth edge",
        ],
        "abnormal": [
            "a zipper with broken or missing teeth in a local region",
            "a zipper with split teeth or squeezed teeth",
            "a zipper with torn or defective fabric border",
            "a zipper with broken fabric near the zipper edge",
            "a zipper with locally misaligned tooth structure",
        ],
    },
}

MVTEC_CLASS_KEYS = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}


def is_mvtec_class_name(class_name):
    class_key = str(class_name or "").replace(" ", "_").strip().lower()
    return class_key in MVTEC_CLASS_KEYS


def build_defect_prompt_variants(class_name, defect_prompts):
    class_name = str(class_name).replace("_", " ").strip()
    base_prompts = dedupe_keep_order(defect_prompts)
    variants = []
    for prompt in base_prompts:
        prompt_text = str(prompt).strip()
        if not prompt_text:
            continue
        variants.extend(
            [
                prompt_text,
                f"a photo of {prompt_text}",
                f"a close-up photo of {prompt_text}",
                f"a zoomed-in photo of {prompt_text}",
                f"a localized defect described as {prompt_text}",
            ]
        )
        if class_name and class_name.lower() not in prompt_text.lower():
            variants.extend(
                [
                    f"a photo of {class_name} showing {prompt_text}",
                    f"a close-up photo of {class_name} showing {prompt_text}",
                    f"a localized anomaly on {class_name}: {prompt_text}",
                ]
            )
    return dedupe_keep_order(variants)


def build_compact_defect_prompt_variants(class_name, defect_prompts):
    class_name = str(class_name).replace("_", " ").strip()
    base_prompts = dedupe_keep_order(defect_prompts)
    variants = []
    for prompt in base_prompts:
        prompt_text = str(prompt).strip()
        if not prompt_text:
            continue
        variants.extend(
            [
                prompt_text,
                f"a photo of {prompt_text}",
                f"a close-up photo of {prompt_text}",
            ]
        )
        if class_name and class_name.lower() not in prompt_text.lower():
            variants.append(f"a close-up photo of a local defect on {class_name}: {prompt_text}")
    return dedupe_keep_order(variants)


def build_prompt_ensemble_lists(class_name, good_prompts, defect_prompts):
    class_name = str(class_name).replace("_", " ").strip()
    if is_mvtec_class_name(class_name):
        extra_good = [
            f"a photo of a normal {class_name}",
            f"a photo of a defect-free {class_name}",
            f"a photo of an intact {class_name}",
            f"a close-up photo of a normal {class_name}",
            f"a close-up photo of a clean {class_name} surface",
        ]
        extra_defect = [
            f"a photo of a local defect on {class_name}",
            f"a close-up photo of a subtle anomaly on {class_name}",
            f"a zoomed-in photo of a small defect region on {class_name}",
            f"a photo of a damaged area on {class_name}",
        ]
        defect_type_prompts = build_compact_defect_prompt_variants(class_name, defect_prompts)
        return (
            dedupe_keep_order([*good_prompts, *extra_good]),
            dedupe_keep_order([*defect_type_prompts, *extra_defect]),
        )

    extra_good = [
        "a photo of a normal object",
        "a photo of a defect-free object",
        "a photo of an intact object",
        "a photo of a clean object",
        "a photo of a normal industrial object",
        "a close-up photo of a normal object",
        f"a photo of a normal {class_name}",
        f"a photo of a defect-free {class_name}",
        f"a close-up photo of a normal {class_name}",
    ]
    extra_defect = [
        "a photo of a defective object",
        "a photo of an abnormal object",
        "a photo of an object with anomaly",
        "a photo of an object with damage",
        "a photo of an object with defect",
        "a close-up photo of a local defect",
        "a zoomed-in photo of an abnormal region",
        "a photo of a scratched object",
        "a photo of a cracked object",
        "a photo of a contaminated object",
        f"a photo of a defective {class_name}",
        f"a photo of a damaged {class_name}",
        f"a close-up photo of a defect on {class_name}",
        f"a zoomed-in photo of an abnormal region on {class_name}",
    ]
    defect_type_prompts = build_defect_prompt_variants(class_name, defect_prompts)
    return (
        dedupe_keep_order([*good_prompts, *extra_good]),
        dedupe_keep_order([*defect_type_prompts, *extra_defect]),
    )


def build_mvtec_sota_prompt_lists(class_name, good_prompts, defect_prompts):
    class_name = str(class_name).replace("_", " ").strip()
    class_key = class_name.replace(" ", "_").lower()
    targeted = MVTec_TARGETED_PROMPTS.get(class_key, {})

    promptad_defect_phrases = list(defect_prompts)
    for defect_name in MVTec_PROMPTAD_STYLE_DEFECTS.get(class_key, []):
        promptad_defect_phrases.extend(
            [
                f"{class_name} with {defect_name}",
                f"{class_name} showing {defect_name}",
            ]
        )

    targeted_normal = list(targeted.get("normal", []))
    targeted_abnormal = dedupe_keep_order([*targeted.get("abnormal", []), *promptad_defect_phrases])

    normal_prompts = []
    for prompt_text in dedupe_keep_order([*good_prompts, *targeted_normal]):
        normal_prompts.extend(
            [
                prompt_text,
                f"a close-up industrial inspection photo of {prompt_text}",
            ]
        )

    abnormal_prompts = []
    for prompt_text in targeted_abnormal:
        abnormal_prompts.extend(
            [
                prompt_text,
                f"a close-up industrial inspection photo of {prompt_text}",
                f"a localized defect on {class_name}: {prompt_text}",
            ]
        )

    spatial_prompts = [
        f"a tiny local defect on {class_name}",
        f"a subtle abnormal region on {class_name}",
        f"a fine-grained surface defect on {class_name}",
    ]

    return (
        dedupe_keep_order(normal_prompts),
        dedupe_keep_order(targeted_abnormal),
        dedupe_keep_order([*abnormal_prompts, *spatial_prompts]),
    )


def build_prompts(
    class_name=None,
    prompt_style="default",
    base_normal_prompts=None,
    base_abnormal_prompts=None,
    enable_prompt_ensemble=False,
):
    normal_prompts = list(base_normal_prompts or [])
    abnormal_prompts = list(base_abnormal_prompts or [])
    class_name = str(class_name).replace("_", " ").strip() if class_name is not None else None

    if not enable_prompt_ensemble:
        normal_prompts = dedupe_keep_order(normal_prompts)
        abnormal_prompts = dedupe_keep_order(abnormal_prompts)
        return {
            "prompt_style": "default",
            "normal_prompts": normal_prompts,
            "general_abnormal_prompts": [],
            "spatial_abnormal_prompts": [],
            "defect_type_prompts": [],
            "abnormal_prompts": abnormal_prompts,
            "prompt_ensemble_enabled": False,
        }

    if prompt_style == "mvtec_sota":
        normal_prompts, defect_type_prompts, abnormal_prompts = build_mvtec_sota_prompt_lists(
            class_name or "",
            normal_prompts,
            abnormal_prompts,
        )
        general_abnormal_prompts = [
            prompt
            for prompt in abnormal_prompts
            if "localized defect on" not in prompt and "small defective region on" not in prompt
        ]
        spatial_abnormal_prompts = (
            [
                f"a tiny defect on {class_name}",
                f"a subtle local anomaly on {class_name}",
                f"a thin scratch-like defect on {class_name}",
                f"a small spot defect on {class_name}",
                f"a fine-grained abnormal region on {class_name}",
            ]
            if class_name
            else []
        )
        abnormal_prompts = dedupe_keep_order([*general_abnormal_prompts, *spatial_abnormal_prompts, *abnormal_prompts])
        return {
            "prompt_style": prompt_style,
            "normal_prompts": normal_prompts,
            "general_abnormal_prompts": dedupe_keep_order(general_abnormal_prompts),
            "spatial_abnormal_prompts": dedupe_keep_order(spatial_abnormal_prompts),
            "defect_type_prompts": defect_type_prompts,
            "abnormal_prompts": abnormal_prompts,
            "prompt_ensemble_enabled": True,
        }

    if prompt_style in {"spatial_aware", "pcb_specific"}:
        if is_mvtec_class_name(class_name):
            normal_prompts = [
                *normal_prompts,
                f"a photo of a normal {class_name}",
                f"a photo of a defect-free {class_name}",
                f"a photo of an intact {class_name}",
                f"a close-up photo of a clean {class_name} surface",
                f"a close-up photo of a regular {class_name} structure",
            ]
            general_abnormal_prompts = [
                f"a photo of a defective {class_name}",
                f"a close-up photo of a damaged {class_name}",
                f"a photo of a local anomaly on {class_name}",
            ]
            spatial_abnormal_prompts = [
                f"a photo of a tiny local defect on {class_name}",
                f"a photo of a thin defect region on {class_name}",
                f"a close-up photo of a subtle anomaly on {class_name}",
                f"a zoomed-in photo of a fine-grained defect on {class_name}",
            ]
            defect_type_prompts = build_compact_defect_prompt_variants(class_name or "", abnormal_prompts)
            generic_defect_type_prompts = [
                f"a close-up photo of a local defect on {class_name}",
                f"a zoomed-in photo of an abnormal region on {class_name}",
            ]
        else:
            normal_prompts = [
                *normal_prompts,
                "a photo of a normal object",
                "a photo of a flawless object",
                "a photo of a defect-free industrial product",
                "a photo of an intact object",
                "a photo of a clean surface",
                "a photo of a uniform texture",
                "a photo of a smooth and undamaged surface",
                "a photo of a normal industrial product",
                "a photo of an object without anomaly",
                "a photo of an object without crack",
                "a photo of an object without scratch",
                "a photo of an object without local defect",
            ]
            general_abnormal_prompts = [
                "a photo of a defective object",
                "a photo of a damaged object",
                "a photo of an abnormal industrial product",
                "a photo of an object with anomaly",
                "a photo of a flawed object",
            ]
            spatial_abnormal_prompts = [
                "a photo of a tiny spot defect",
                "a photo of a small dot defect",
                "a photo of a thin line defect",
                "a photo of a small localized defect",
                "a photo of an irregular patch defect",
                "a photo of a local texture anomaly",
                "a photo of a subtle local anomaly",
                "a photo of a fine-grained defect region",
            ]
            defect_type_prompts = build_defect_prompt_variants(class_name or "", abnormal_prompts)
            generic_defect_type_prompts = [
                "a photo of a crack",
                "a photo of a scratch",
                "a photo of contamination",
                "a photo of a stain",
                "a photo of a broken surface",
                "a photo of a missing part",
                "a photo of deformation",
                "a photo of a damaged surface",
                "a close-up photo of a local defect",
                "a zoomed-in photo of a fine-grained anomaly",
            ]
        if class_name:
            if not is_mvtec_class_name(class_name):
                normal_prompts.extend(
                    [
                        f"a photo of a normal {class_name}",
                        f"a photo of a flawless {class_name}",
                    ]
                )
                general_abnormal_prompts.extend(
                    [
                        f"a photo of a defective {class_name}",
                        f"a photo of a damaged {class_name}",
                    ]
                )
                spatial_abnormal_prompts.extend(
                    [
                        f"a photo of a {class_name} with a tiny spot defect",
                        f"a photo of a {class_name} with a thin line defect",
                        f"a photo of a {class_name} with a local texture anomaly",
                    ]
                )
        if prompt_style == "pcb_specific":
            pcb_normal_prompts = []
            pcb_abnormal_prompts = []
            class_key = (class_name or "").lower()
            if "pcb" in class_key or "printed circuit" in class_key:
                pcb_normal_prompts = [
                    f"a photo of a normal {class_name} printed circuit board",
                    f"a close-up photo of an intact {class_name} circuit board",
                    f"a photo of a clean {class_name} pcb surface",
                    f"a photo of {class_name} with complete electronic components",
                    f"a photo of {class_name} with normal solder joints",
                    f"a photo of {class_name} with intact copper traces",
                ]
                pcb_abnormal_prompts = [
                    f"a photo of {class_name} with a missing electronic component",
                    f"a photo of {class_name} with a bent electronic component",
                    f"a photo of {class_name} with a shifted component",
                    f"a photo of {class_name} with a broken solder joint",
                    f"a photo of {class_name} with solder contamination",
                    f"a photo of {class_name} with a scratched copper trace",
                    f"a photo of {class_name} with a tiny spot defect on the pcb",
                    f"a photo of {class_name} with a small dot defect on the circuit board",
                    f"a photo of {class_name} with a thin line scratch on the pcb",
                    f"a photo of {class_name} with a local defect near an electronic component",
                    f"a close-up photo of a tiny abnormal region on {class_name}",
                    f"a zoomed-in photo of a fine-grained pcb defect on {class_name}",
                ]
            normal_prompts.extend(pcb_normal_prompts)
            spatial_abnormal_prompts.extend(pcb_abnormal_prompts)
        normal_prompts = dedupe_keep_order(normal_prompts)
        general_abnormal_prompts = dedupe_keep_order(general_abnormal_prompts)
        spatial_abnormal_prompts = dedupe_keep_order(spatial_abnormal_prompts)
        defect_type_prompts = dedupe_keep_order([*defect_type_prompts, *generic_defect_type_prompts])
        abnormal_prompts = dedupe_keep_order(
            [*general_abnormal_prompts, *spatial_abnormal_prompts, *defect_type_prompts]
        )
        return {
            "prompt_style": prompt_style,
            "normal_prompts": normal_prompts,
            "general_abnormal_prompts": general_abnormal_prompts,
            "spatial_abnormal_prompts": spatial_abnormal_prompts,
            "defect_type_prompts": defect_type_prompts,
            "abnormal_prompts": abnormal_prompts,
            "prompt_ensemble_enabled": True,
        }

    normal_prompts, abnormal_prompts = build_prompt_ensemble_lists(class_name or "", normal_prompts, abnormal_prompts)
    return {
        "prompt_style": "default",
        "normal_prompts": normal_prompts,
        "general_abnormal_prompts": [],
        "spatial_abnormal_prompts": [],
        "defect_type_prompts": [],
        "abnormal_prompts": abnormal_prompts,
        "prompt_ensemble_enabled": True,
    }


def load_adaptive_prompt_policy(policy_path):
    if policy_path is None:
        return None
    policy_path = Path(policy_path)
    if not policy_path.exists():
        print(f"[WARN] adaptive prompt policy not found, fallback to default: {policy_path}")
        return None
    with open(policy_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "adaptive_prompt_policy" in payload and isinstance(payload["adaptive_prompt_policy"], dict):
        return payload["adaptive_prompt_policy"]
    return payload


def resolve_prompt_style_for_class(dataset, class_name, requested_prompt_style, adaptive_policy=None):
    dataset = str(dataset or "").strip().lower()
    class_name_norm = str(class_name or "").replace("_", " ").strip().lower()
    selected_prompt_style = requested_prompt_style
    adaptive_policy_name = None

    if requested_prompt_style != "adaptive":
        return {
            "requested_prompt_style": requested_prompt_style,
            "selected_prompt_style": selected_prompt_style,
            "adaptive_policy_name": adaptive_policy_name,
        }

    selected_prompt_style = "default"
    dataset_policy = adaptive_policy.get(dataset) if isinstance(adaptive_policy, dict) else None
    if not isinstance(dataset_policy, dict):
        return {
            "requested_prompt_style": requested_prompt_style,
            "selected_prompt_style": selected_prompt_style,
            "adaptive_policy_name": adaptive_policy_name,
        }

    adaptive_policy_name = dataset_policy.get("policy_name")
    spatial_aware_classes = dataset_policy.get("spatial_aware", [])
    default_classes = dataset_policy.get("default", [])

    spatial_aware_set = {
        str(name).replace("_", " ").strip().lower()
        for name in spatial_aware_classes
        if str(name).strip()
    }
    if class_name_norm in spatial_aware_set:
        selected_prompt_style = "spatial_aware"
    elif isinstance(default_classes, str) and default_classes.strip().lower() == "all":
        selected_prompt_style = "default"
    else:
        default_set = {
            str(name).replace("_", " ").strip().lower()
            for name in (default_classes or [])
            if str(name).strip()
        }
        if class_name_norm in default_set:
            selected_prompt_style = "default"

    return {
        "requested_prompt_style": requested_prompt_style,
        "selected_prompt_style": selected_prompt_style,
        "adaptive_policy_name": adaptive_policy_name,
    }


def save_prompt_stats(output_dir, class_name, prompt_bundle):
    prompt_stats_dir = Path(output_dir) / "prompt_stats"
    prompt_stats_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_name": class_name,
        "requested_prompt_style": prompt_bundle.get("requested_prompt_style", prompt_bundle["prompt_style"]),
        "prompt_style": prompt_bundle["prompt_style"],
        "adaptive_selected_prompt_style": prompt_bundle.get("adaptive_selected_prompt_style", prompt_bundle["prompt_style"]),
        "adaptive_policy_name": prompt_bundle.get("adaptive_policy_name"),
        "prompt_ensemble_enabled": bool(prompt_bundle.get("prompt_ensemble_enabled", False)),
        "num_normal_prompts": len(prompt_bundle["normal_prompts"]),
        "num_general_abnormal_prompts": len(prompt_bundle["general_abnormal_prompts"]),
        "num_spatial_abnormal_prompts": len(prompt_bundle["spatial_abnormal_prompts"]),
        "num_defect_type_prompts": len(prompt_bundle["defect_type_prompts"]),
        "num_abnormal_prompts": len(prompt_bundle["abnormal_prompts"]),
        "normal_prompts": prompt_bundle["normal_prompts"],
        "general_abnormal_prompts": prompt_bundle["general_abnormal_prompts"],
        "spatial_abnormal_prompts": prompt_bundle["spatial_abnormal_prompts"],
        "defect_type_prompts": prompt_bundle["defect_type_prompts"],
        "abnormal_prompts": prompt_bundle["abnormal_prompts"],
    }
    out_path = prompt_stats_dir / f"{make_relative_safe_name(str(class_name))}_prompts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


@torch.inference_mode()
def encode_prompt_ensemble(tokenizer, model, prompts):
    prompt_features = encode_text(prompts, tokenizer, model)
    return average_prompt_features(prompt_features)


def average_prompt_features(prompt_features):
    feat_mean = prompt_features.mean(dim=0, keepdim=True)
    feat_mean = feat_mean / feat_mean.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feat_mean


def apply_final_flag_overrides(args):
    method_name = str(getattr(args, "resolved_method_name", "") or getattr(args, "method_name", "") or "").lower()
    if "ncrs" in method_name:
        args.normal_calibrated_selection = True
        args.pars_save_selection_stats = True

    if getattr(args, "_user_disabled_fg", False):
        args.enable_fg = False
        args.foreground_mode = "none"
        args.global_guided_fusion = False

    if not getattr(args, "normal_calibrated_selection", False):
        args.normal_calibrated_selection = False
    if not getattr(args, "pars_save_selection_stats", False):
        args.pars_save_selection_stats = False
    if not getattr(args, "response_calibration", False):
        args.response_calibration = False
    if not getattr(args, "reliability_ms_fusion", False):
        args.reliability_ms_fusion = False
    return args


def resolve_ablation_flags(args):
    if args.method in HPMR_METHODS and (
        getattr(args, "_user_requested_fg", False) or getattr(args, "_user_requested_foreground_mode", False)
    ):
        warnings.warn("FG/foreground module is disabled for HPMR-CLIP experiments.")

    if args.method in LEGACY_METHODS:
        legacy_presets = {
            "baseline": (False, False, False, False, False),
            "fgclip_baseline": (False, False, False, False, False),
            MG_METHOD_NO_REFINER: (True, False, False, False, False),
            MG_METHOD_REFINER: (True, False, False, False, False),
            "mg_positive": (True, False, False, False, False),
            "mg_only": (True, False, False, False, False),
            "mg_fg": (True, True, False, False, False),
            "mg_ms": (True, False, True, False, False),
            "mg_fg_ms": (True, True, True, False, False),
            MS_METHOD: (True, True, True, True, False),
            "mg_ms_hybrid_prompt": (True, True, True, True, True),
            "full_ours": (True, True, True, True, False),
        }
        preset = legacy_presets.get(args.method)
        if preset is not None:
            if args.enable_mg is None:
                args.enable_mg = preset[0]
            if args.enable_fg is None:
                args.enable_fg = preset[1]
            if args.enable_ms is None:
                args.enable_ms = preset[2]
            if args.enable_positive_fusion is None:
                args.enable_positive_fusion = preset[3]
            if not getattr(args, "prompt_ensemble", False):
                args.prompt_ensemble = preset[4]
        if args.method_name in {"mg_ms_hybrid_prompt", "mg_ms_hybrid_prompt_ncrs"}:
            args.prompt_ensemble = True

        if args.enable_mg is None:
            args.enable_mg = True
        if args.enable_fg is None:
            args.enable_fg = True
        if args.enable_ms is None:
            args.enable_ms = True
        if args.enable_positive_fusion is None:
            args.enable_positive_fusion = args.mg_fusion_mode == "positive"
        if not hasattr(args, "response_calibration") or args.response_calibration is None:
            args.response_calibration = False
        if not hasattr(args, "text_conditioned_mask") or args.text_conditioned_mask is None:
            args.text_conditioned_mask = False
        if not hasattr(args, "global_guided_fusion") or args.global_guided_fusion is None:
            args.global_guided_fusion = False

        if not args.enable_mg:
            args.enable_ms = False
            args.enable_positive_fusion = False
            args.mg_fusion_mode = "off"
        elif args.enable_positive_fusion:
            args.mg_fusion_mode = "positive"
        elif args.mg_fusion_mode == "off":
            args.mg_fusion_mode = "direct"

        if args.method_name:
            args.resolved_method_name = args.method_name
        elif getattr(args, "normal_calibrated_selection", False) and args.method == MS_METHOD:
            args.resolved_method_name = "mg_ms_hybrid_prompt_ncrs"
        else:
            args.resolved_method_name = args.method
        return apply_final_flag_overrides(args)

    presets = {
        "fgclip_baseline": {
            "prompt_ensemble": False,
            "enable_mg": False,
            "enable_ms": False,
            "enable_positive_fusion": False,
            "response_calibration": False,
            "text_conditioned_mask": False,
        },
        "prompt": {
            "prompt_ensemble": True,
            "enable_mg": False,
            "enable_ms": False,
            "enable_positive_fusion": False,
            "response_calibration": False,
            "text_conditioned_mask": False,
        },
        "prompt_mask": {
            "prompt_ensemble": True,
            "enable_mg": True,
            "enable_ms": False,
            "enable_positive_fusion": True,
            "response_calibration": False,
            "text_conditioned_mask": True,
        },
        "prompt_mask_ms": {
            "prompt_ensemble": True,
            "enable_mg": True,
            "enable_ms": True,
            "enable_positive_fusion": True,
            "response_calibration": False,
            "text_conditioned_mask": True,
        },
        "full": {
            "prompt_ensemble": True,
            "enable_mg": True,
            "enable_ms": True,
            "enable_positive_fusion": True,
            "response_calibration": False,
            "text_conditioned_mask": True,
        },
    }
    preset = presets[args.method]
    args.prompt_ensemble = preset["prompt_ensemble"]
    args.enable_mg = preset["enable_mg"]
    args.enable_fg = False
    args.enable_ms = preset["enable_ms"]
    args.enable_positive_fusion = preset["enable_positive_fusion"]
    args.response_calibration = preset["response_calibration"]
    args.text_conditioned_mask = preset["text_conditioned_mask"]
    args.global_guided_fusion = False
    args.foreground_mode = "none"

    if not args.enable_mg:
        args.enable_fg = False
        args.enable_ms = False
        args.enable_positive_fusion = False
        args.mg_fusion_mode = "off"
    elif args.enable_positive_fusion:
        args.mg_fusion_mode = "positive"
    elif args.mg_fusion_mode == "off":
        args.mg_fusion_mode = "direct"

    if args.method_name:
        args.resolved_method_name = args.method_name
    elif args.method in HPMR_METHODS:
        args.resolved_method_name = args.method
    else:
        args.resolved_method_name = (
            f"mg{int(args.enable_mg)}_fg{int(args.enable_fg)}_ms{int(args.enable_ms)}_pf{int(args.enable_positive_fusion)}"
        )
    return apply_final_flag_overrides(args)


def pixel_ap(scores, labels):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    num_pos = int(np.sum(labels == 1))
    if num_pos == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    pos_positions = np.flatnonzero(sorted_labels == 1)
    numerator = np.arange(1, num_pos + 1, dtype=np.float32)
    denominator = pos_positions.astype(np.float32, copy=False) + 1.0
    return float(np.sum(numerator / denominator, dtype=np.float64) / num_pos)


def compute_pixel_metrics(scores, labels, exact_max_pixels=DEFAULT_METRIC_EXACT_MAX_PIXELS, bins=DEFAULT_METRIC_BINS):
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if scores.size <= int(exact_max_pixels):
        return compute_pixel_metrics_exact(scores, labels)
    print(
        f"[WARN] Pixel metric arrays contain {scores.size} pixels; "
        f"using {bins}-bin histogram approximation to avoid OOM."
    )
    return compute_pixel_metrics_binned(scores, labels, bins)


def compute_pixel_metrics_exact(scores, labels):
    num_pos = int(np.sum(labels == 1))
    num_neg = int(np.sum(labels == 0))
    if num_pos == 0 or num_neg == 0:
        return {"seg_auc": float("nan"), "seg_f1": 0.0, "pixel_ap": float("nan")}

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    distinct = np.empty(sorted_scores.shape[0], dtype=bool)
    distinct[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    distinct[-1] = True
    distinct_idx = np.flatnonzero(distinct)
    tp_cum = np.cumsum(sorted_labels, dtype=np.int64)

    tp = tp_cum[distinct_idx].astype(np.float32, copy=False)
    selected = (distinct_idx + 1).astype(np.float32, copy=False)
    fp = selected - tp
    tpr = np.concatenate(([0.0], tp / float(num_pos), [1.0])).astype(np.float32, copy=False)
    fpr = np.concatenate(([0.0], fp / float(num_neg), [1.0])).astype(np.float32, copy=False)
    seg_auc = float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]), dtype=np.float64))

    precision = np.divide(tp, selected, out=np.zeros_like(tp), where=selected > 0)
    recall = tp / float(num_pos)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    seg_f1 = float(f1[int(np.argmax(f1))])

    pos_positions = np.flatnonzero(sorted_labels == 1)
    numerator = np.arange(1, num_pos + 1, dtype=np.float32)
    denominator = pos_positions.astype(np.float32, copy=False) + 1.0
    ap = float(np.sum(numerator / denominator, dtype=np.float64) / num_pos)
    return {"seg_auc": seg_auc, "seg_f1": seg_f1, "pixel_ap": ap}


def compute_pixel_metrics_binned(scores, labels, bins):
    num_pos = int(np.sum(labels == 1))
    num_neg = int(np.sum(labels == 0))
    if num_pos == 0 or num_neg == 0:
        return {"seg_auc": float("nan"), "seg_f1": 0.0, "pixel_ap": float("nan")}

    bins = max(256, int(bins))
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return {"seg_auc": float("nan"), "seg_f1": 0.0, "pixel_ap": float("nan")}

    scaled = np.floor((scores - lo) * ((bins - 1) / (hi - lo))).astype(np.int32, copy=False)
    np.clip(scaled, 0, bins - 1, out=scaled)
    pos_hist = np.bincount(scaled, weights=labels, minlength=bins).astype(np.float64, copy=False)
    neg_hist = np.bincount(scaled, weights=(1 - labels), minlength=bins).astype(np.float64, copy=False)

    tp = np.cumsum(pos_hist[::-1])
    fp = np.cumsum(neg_hist[::-1])
    tpr = np.r_[0.0, tp / num_pos, 1.0]
    fpr = np.r_[0.0, fp / num_neg, 1.0]
    seg_auc = float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])))

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / num_pos
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    seg_f1 = float(np.max(f1))
    ap = float(np.sum(precision * pos_hist[::-1]) / num_pos)
    return {"seg_auc": seg_auc, "seg_f1": seg_f1, "pixel_ap": ap}


def make_runtime_args(args, use_mg=False, mg_refiner_dir=None, mg_refiner_checkpoint=None, feature_layer=0):
    runtime_mg_mode = args.mg_fusion_mode
    if runtime_mg_mode == "direct":
        runtime_mg_mode = "linear"
    elif runtime_mg_mode == "off":
        runtime_mg_mode = "none"
    return SimpleNamespace(
        resize_short_edge=args.resize_short_edge,
        max_num_patches=args.max_num_patches,
        feature_layer=int(feature_layer or 0),
        topk_ratio=0.002,
        good_bank_max_patches=GOOD_BANK_MAX_PATCHES,
        good_bank_patches_per_image=GOOD_BANK_PATCHES_PER_IMAGE,
        good_bank_chunk_size=GOOD_BANK_CHUNK_SIZE,
        defect_text_weight=1.0,
        good_text_weight=0.15,
        proto_distance_weight=0.35,
        good_bank_distance_weight=0.85,
        threshold_std_mult=THRESHOLD_STD_MULT,
        min_threshold_margin=MIN_THRESHOLD_MARGIN,
        use_mg=use_mg,
        mg_fusion_mode=runtime_mg_mode if use_mg else "none",
        mg_fusion_weight=args.mg_fusion_weight,
        mg_mask_mode="ratio",
        mg_mask_ratio=args.mg_mask_ratio,
        mg_mask_dilate_radius=1,
        mg_adaptive_k=1.0,
        mg_adaptive_quantile=0.90,
        mg_use_gate=False,
        mg_gate_min_area=0.001,
        mg_gate_max_area=0.3,
        mg_local_only=True,
        mg_start_layer=args.mg_start_layer,
        mg_end_layer=args.mg_end_layer,
        mg_mask_threshold=args.mg_mask_threshold,
        mg_neg_bias=args.mg_neg_bias,
        attention_bias_eta=args.attention_bias_eta,
        mg_refiner_checkpoint=mg_refiner_checkpoint,
        mg_refiner_dir=mg_refiner_dir,
        mg_refiner_threshold=0.5,
        mg_refiner_mask_mode="score_intersect",
        mg_refiner_topk_ratio=0.05,
        mg_refiner_score_ratio=0.08,
        mg_refiner_soft_fusion=True,
        mg_refiner_score_boost=0.0,
        mg_refiner_score_power=1.0,
        mg_use_refiner_weight=False,
        segad_calibrator=None,
        segad_calibrator_dir=None,
        segad_blend_weight=0.0,
        segad_fusion_mode="positive",
        segad_power=1.0,
        segad_min_confidence=0.55,
        image_score_source="final",
        classification_score_source="image_score",
        map_refine_mode="none",
        map_refine_alpha=0.0,
        map_refine_bg_sigma=7.0,
        map_refine_topk_ratio=0.03,
        map_refine_clamp_quantile=1.0,
    )


def build_dual_layer_mask_seed(A_semantic, A_local, args):
    semantic = normalize_map(A_semantic)
    local = normalize_map(A_local)
    source = str(getattr(args, "local_mask_source", "fused") or "fused").strip().lower()
    if source == "local":
        seed = local
    elif source == "intersect":
        seed = semantic * local
    else:
        seed = float(args.semantic_mask_weight) * semantic + float(args.local_mask_weight) * local
    return normalize_map(seed.astype(np.float32)), semantic, local


def get_good_dir(dataset, data_root, class_name, train_samples):
    if dataset == "mvtec":
        return data_root / class_name / "train" / "good"
    if not train_samples:
        raise RuntimeError(f"No VisA train samples found for {class_name}")
    return train_samples[0].image_path.parents[2]


def get_text_banks(dataset, class_name, tokenizer, model):
    if dataset == "visa":
        good_prompts, defect_prompts = get_visa_prompts(class_name)
    else:
        good_prompts, defect_prompts = get_mvtec_prompts(class_name)
    return good_prompts, defect_prompts


def build_text_feature_bundle(tokenizer, model, prompt_bundle):
    normal_prompts = prompt_bundle["normal_prompts"]
    abnormal_prompts = prompt_bundle["abnormal_prompts"]
    if prompt_bundle.get("prompt_ensemble_enabled", False):
        normal_prompt_bank = encode_text(normal_prompts, tokenizer, model)
        abnormal_prompt_bank = encode_text(abnormal_prompts, tokenizer, model)
        return SimpleNamespace(
            scoring_good_text=average_prompt_features(normal_prompt_bank),
            scoring_defect_text=average_prompt_features(abnormal_prompt_bank),
            mask_good_text=normal_prompt_bank,
            mask_defect_text=abnormal_prompt_bank,
            good_prompts=normal_prompts,
            defect_prompts=abnormal_prompts,
        )
    normal_prompt_bank = encode_text(normal_prompts, tokenizer, model)
    abnormal_prompt_bank = encode_text(abnormal_prompts, tokenizer, model)
    return SimpleNamespace(
        scoring_good_text=normal_prompt_bank,
        scoring_defect_text=abnormal_prompt_bank,
        mask_good_text=normal_prompt_bank,
        mask_defect_text=abnormal_prompt_bank,
        good_prompts=normal_prompts,
        defect_prompts=abnormal_prompts,
    )


def build_inference_context(image_processor, model, good_proto, good_memory, text_features, runtime_args):
    return SimpleNamespace(
        image_processor=image_processor,
        model=model,
        good_proto=good_proto,
        good_memory=good_memory,
        scoring_good_text=text_features.scoring_good_text,
        scoring_defect_text=text_features.scoring_defect_text,
        mask_good_text=text_features.mask_good_text,
        mask_defect_text=text_features.mask_defect_text,
        runtime_args=runtime_args,
    )


def infer_single_map(image, ctx):
    raw_map, _, score = compute_maps_and_score(
        image=image,
        image_processor=ctx.image_processor,
        model=ctx.model,
        good_proto=ctx.good_proto,
        good_memory_bank=ctx.good_memory,
        good_text_bank=ctx.scoring_good_text,
        defect_text_bank=ctx.scoring_defect_text,
        args=ctx.runtime_args,
    )
    return raw_map.astype(np.float32), float(score)


def compute_heatmap_image_score(heatmap, topk_ratio):
    array = ensure_finite_array("image_score_heatmap", heatmap)
    flat = array.reshape(-1)
    keep = max(1, int(len(flat) * max(float(topk_ratio), 1e-6)))
    return float(np.mean(np.sort(flat)[-keep:]))


def infer_text_conditioned_outputs(image, ctx, args, mask_base_heatmap=None):
    patch_features, real_h, real_w, _ = encode_dense_image(
        image=image,
        image_processor=ctx.image_processor,
        model=ctx.model,
        resize_target=ctx.runtime_args.resize_short_edge,
        max_num_patches=ctx.runtime_args.max_num_patches,
    )
    text_score = compute_text_conditioned_score(
        patch_features=patch_features,
        normal_text_features=ctx.mask_good_text,
        abnormal_text_features=ctx.mask_defect_text,
    )
    text_conditioned_heatmap = patch_score_to_heatmap(
        text_score=text_score.detach().cpu().numpy(),
        image_size=image.size,
        patch_grid_shape=(real_h, real_w),
    )
    text_mask_input = normalize_map(text_conditioned_heatmap)
    if mask_base_heatmap is not None:
        text_mask_input = normalize_map(text_mask_input * normalize_map(mask_base_heatmap))
    text_mask = compute_text_conditioned_mask(
        text_conditioned_heatmap=text_mask_input,
        tau=args.text_mask_tau,
        beta=args.text_mask_beta,
    )
    psme_heatmap = apply_text_conditioned_psme(
        text_conditioned_heatmap=text_conditioned_heatmap,
        text_mask=text_mask,
        lambda_weight=args.lambda_weight,
    )
    return {
        "text_score": ensure_finite_array("text_score", text_score.detach().cpu().numpy()),
        "text_conditioned_heatmap": text_conditioned_heatmap,
        "text_mask_input": text_mask_input,
        "text_mask": text_mask,
        "psme_heatmap": psme_heatmap,
    }


def save_intermediate_outputs(output_dir, image, dataset, class_name, sample, debug):
    sample_name = make_relative_safe_name(
        f"{dataset}_{class_name}_{sample.split}_{sample.label_name}_{sample.image_path.stem}"
    )
    intermediate_root = Path(output_dir) / "intermediate"
    stage_to_value = {
        "baseline": debug.get("baseline_heatmap"),
        "text_conditioned_heatmap": debug.get("text_conditioned_heatmap"),
        "text_mask": debug.get("text_mask"),
        "psme_heatmap": debug.get("psme_heatmap"),
        "local_weighted_heatmap": debug.get("local_weighted_heatmap"),
        "final_heatmap": debug.get("final_heatmap"),
    }
    for stage_name, value in stage_to_value.items():
        if value is None:
            continue
        save_intermediate_heatmap(
            output_dir=intermediate_root,
            stage_name=stage_name,
            sample_stem=sample_name,
            heatmap=value,
            image=image,
        )


def infer_ms_fb_mg(image, baseline_ctx, mg_ctx, baseline_stats, mg_stats, args, ncrs_tau_risk=None, local_ctx=None, local_stats=None):
    use_hpmr_mode = args.method in HPMR_METHODS
    base_map_raw, baseline_score = infer_single_map(image, baseline_ctx)
    width, height = image.size

    # Keep baseline/MG evaluation maps aligned with run_mvtec_ad_benchmark.py:
    # raw patch map -> train-good standardization -> BICUBIC resize. Do not
    # per-image normalize these maps before pixel-level metrics.
    A_baseline = ensure_finite_array(
        "baseline_heatmap",
        resize_map(standardize_map(base_map_raw, *baseline_stats), (width, height)).astype(np.float32),
    )
    mg_score = baseline_score
    A_full = A_baseline
    A_tile = A_baseline
    local_weighted_heatmap = None
    M_fg = np.ones((height, width), dtype=np.float32)
    tile_count = 0
    text_score = None
    text_conditioned_heatmap = None
    text_mask_input = None
    text_mask = None
    psme_heatmap = None
    global_heatmap_norm = None
    local_heatmap_norm = None
    tile_guidance_scores = None
    tile_guidance_scores_raw = None
    tile_weights = None
    reliability_stats = None
    selection_stats = None
    iterative_refinement_trace = []
    global_ctx = mg_ctx if args.enable_mg else baseline_ctx
    A_semantic = A_baseline
    A_local = None
    A_mask_seed = None
    mask_source_map = A_baseline
    candidate_mask_source_map = A_baseline
    candidate_mask_source_name = "baseline"

    if args.enable_mg:
        full_raw, mg_score = infer_single_map(image, mg_ctx)
        A_full = ensure_finite_array(
            "global_heatmap",
            resize_map(standardize_map(full_raw, *mg_stats), (width, height)).astype(np.float32),
        )
        if use_hpmr_mode:
            A_full = normalize_map(A_full)
    A_semantic = A_full

    if getattr(args, "dual_layer_guidance", False):
        try:
            if local_ctx is None or local_stats is None:
                raise RuntimeError("local_ctx/local_stats missing")
            local_raw, _ = infer_single_map(image, local_ctx)
            A_local = ensure_finite_array(
                "local_heatmap_dual",
                resize_map(standardize_map(local_raw, *local_stats), (width, height)).astype(np.float32),
            )
            if use_hpmr_mode:
                A_local = normalize_map(A_local)
        except Exception as exc:
            if not getattr(args, "_dual_layer_warning_printed", False):
                print(f"[WARN] dual_layer_guidance failed, falling back to semantic guidance only. Reason: {exc}")
                args._dual_layer_warning_printed = True
            A_local = np.asarray(A_semantic, dtype=np.float32).copy()
        A_mask_seed, _, _ = build_dual_layer_mask_seed(A_semantic, A_local, args)
        mask_source_map = A_mask_seed
    else:
        mask_source_map = A_full

    if args.text_conditioned_mask:
        text_outputs = infer_text_conditioned_outputs(
            image,
            global_ctx,
            args,
            mask_base_heatmap=mask_source_map if getattr(args, "dual_layer_guidance", False) else None,
        )
        text_score = text_outputs["text_score"]
        text_conditioned_heatmap = normalize_map(text_outputs["text_conditioned_heatmap"])
        text_mask_input = normalize_map(text_outputs["text_mask_input"])
        text_mask = text_outputs["text_mask"]
        psme_heatmap = normalize_map(text_outputs["psme_heatmap"])
        A_full = psme_heatmap
        mg_score = baseline_score
        if getattr(args, "dual_layer_guidance", False):
            mask_source_map = A_mask_seed if A_mask_seed is not None else mask_source_map
    if args.enable_mg:
        A_full_norm = normalize_map(mask_source_map if getattr(args, "dual_layer_guidance", False) else A_full)
        if args.enable_fg:
            M_fg = make_foreground_mask(
                image,
                A_full_norm,
                mode=args.foreground_mode,
                heatmap_quantile=args.foreground_heatmap_quantile,
                min_area=args.foreground_min_area,
                max_area=args.foreground_max_area,
            )
        if args.enable_ms:
            tiles = make_overlapping_tiles(image, mode=args.tile_mode, overlap=args.overlap)
            tile_items = []
            for tile in tiles:
                tile_raw, _ = infer_single_map(tile["image"], mg_ctx)
                tile_eval = standardize_map(tile_raw, *mg_stats)
                if use_hpmr_mode:
                    tile_eval = normalize_map(tile_eval)
                tile_items.append({**tile, "heatmap": tile_eval})
            if args.global_guided_fusion:
                fusion = global_guided_multiscale_fusion(
                    global_heatmap=A_full,
                    tile_heatmaps=[item["heatmap"] for item in tile_items],
                    tile_boxes=tile_items,
                    alpha=args.alpha,
                    topk_ratio=args.tile_topk_ratio,
                    temperature=args.tile_weight_temperature,
                )
                tile_guidance_scores_raw = fusion["tile_scores_raw"]
                tile_guidance_scores = fusion["tile_scores"]
                tile_weights = fusion["tile_weights"]
                global_heatmap_norm = fusion["global_heatmap_norm"]
                local_weighted_heatmap = fusion["local_weighted_heatmap"]
                local_heatmap_norm = fusion["local_heatmap_norm"]
                A_tile = local_weighted_heatmap
            else:
                A_tile = paste_tile_heatmaps(tile_items, height, width).astype(np.float32)
                if use_hpmr_mode:
                    A_tile = normalize_map(A_tile)
            tile_count = len(tiles)

    mg_mask = None
    mg_mask_threshold = None
    if args.enable_mg:
        candidate_mask_source_map, candidate_mask_source_name = select_candidate_mask_source_heatmap(
            args,
            default_heatmap=mask_source_map,
            tile_heatmap=A_tile if args.enable_ms else None,
            text_heatmap=text_conditioned_heatmap,
        )
        mg_mask, mg_mask_threshold = build_candidate_mask_from_heatmap(
            candidate_mask_source_map,
            args,
        )
        if mg_mask.shape != A_full.shape:
            raise ValueError(f"MG mask shape mismatch: mask={mg_mask.shape}, heatmap={A_full.shape}")

    if not args.enable_mg:
        selected_map = A_full if args.text_conditioned_mask else A_baseline
        selected_score = mg_score if args.text_conditioned_mask else baseline_score
    elif args.enable_ms and args.global_guided_fusion:
        if local_weighted_heatmap is None:
            raise ValueError("global_guided_fusion requires local weighted heatmap when MS branch is enabled.")
        if global_heatmap_norm is None:
            global_heatmap_norm = robust_normalize_heatmap(A_full)
        if local_heatmap_norm is None:
            local_heatmap_norm = robust_normalize_heatmap(local_weighted_heatmap)
        selected_map = ensure_finite_array(
            "final_heatmap",
            ((1.0 - float(args.alpha)) * global_heatmap_norm + float(args.alpha) * local_heatmap_norm).astype(np.float32),
        )
        selected_score = baseline_score
    elif args.enable_fg and not args.enable_ms and not args.enable_positive_fusion:
        selected_map = high_confidence_fusion(A_full, A_full, M_fg, lambda_weight=args.lambda_weight, q=args.q)
        selected_score = mg_score
    elif args.enable_ms and not args.enable_positive_fusion:
        foreground_mask = M_fg if args.enable_fg else None
        if int(getattr(args, "iterative_refinement_steps", 1)) > 1:
            selected_map, mg_mask, iterative_refinement_trace = iterative_refine_multiscale_fusion(
                A_full,
                A_tile,
                args,
                foreground_mask=foreground_mask,
                initial_candidate_mask=mg_mask,
            )
        else:
            fusion_mask = compose_fusion_mask(foreground_mask, mg_mask, args.mg_mask_type)
            selected_map = direct_fusion(A_full, A_tile, lambda_weight=args.lambda_weight, mask=fusion_mask)
        if use_hpmr_mode:
            selected_map = normalize_map(selected_map)
        selected_score = baseline_score
    elif args.enable_ms and args.enable_positive_fusion:
        foreground_mask = M_fg if args.enable_fg else None
        if getattr(args, "reliability_ms_fusion", False):
            fusion_mask = compose_fusion_mask(foreground_mask, mg_mask, args.mg_mask_type)
            selected_map, reliability_stats = reliability_aware_positive_fusion(
                mask_heatmap=A_full,
                tile_heatmap=A_tile,
                candidate_mask=fusion_mask,
                beta=getattr(args, "reliability_beta", 0.08),
                topk_ratio=getattr(args, "reliability_topk_ratio", 0.05),
                min_weight=getattr(args, "reliability_min_weight", 0.0),
                max_weight=getattr(args, "reliability_max_weight", 1.0),
                return_details=True,
            )
        elif int(getattr(args, "iterative_refinement_steps", 1)) > 1:
            selected_map, mg_mask, iterative_refinement_trace = iterative_refine_multiscale_fusion(
                A_full,
                A_tile,
                args,
                foreground_mask=foreground_mask,
                initial_candidate_mask=mg_mask,
            )
        else:
            fusion_mask = compose_fusion_mask(foreground_mask, mg_mask, args.mg_mask_type)
            selected_map = positive_residual_fusion(A_full, A_tile, lambda_weight=args.lambda_weight, mask=fusion_mask, q=args.q)
        if use_hpmr_mode:
            selected_map = normalize_map(selected_map)
        selected_score = baseline_score
    else:
        selected_map = A_full
        selected_score = mg_score

    if getattr(args, "normal_calibrated_selection", False):
        selection_candidate_mask = mg_mask if mg_mask is not None else None
        selected_map, selected_source, q_base, q_enh, r_enh, tau_r = ncrs_select_response(
            baseline_heatmap=A_baseline,
            enhanced_heatmap=selected_map,
            candidate_mask=selection_candidate_mask,
            tau_risk=ncrs_tau_risk,
            margin=getattr(args, "pars_margin", 0.0),
            topk_ratio=getattr(args, "pars_topk_ratio", 0.05),
            area_q=getattr(args, "pars_area_q", 0.80),
        )
        selection_stats = {
            "selected_source": selected_source,
            "q_base": float(q_base),
            "q_enh": float(q_enh),
            "r_enh": float(r_enh),
            "tau_risk": float(tau_r),
            "image_score_raw": float(selected_score),
        }

    if getattr(args, "safe_response_selection", False):
        selection_candidate_mask = None
        if args.enable_ms or args.enable_positive_fusion:
            selection_candidate_mask = mg_mask
        selected_map, selected_source, q_base, q_enh, q_base_stats, q_enh_stats = safe_select_response(
            baseline_heatmap=A_baseline,
            enhanced_heatmap=selected_map,
            candidate_mask=selection_candidate_mask,
            margin=getattr(args, "pars_margin", 0.0),
            topk_ratio=getattr(args, "safe_selection_topk_ratio", 0.05),
            enhanced_bias=getattr(args, "pars_enhanced_bias", 0.0),
        )
        selection_stats = {
            "selected_source": selected_source,
            "q_base": float(q_base),
            "q_enh": float(q_enh),
            "q_base_focus": float(q_base_stats["focus"]),
            "q_base_compactness": float(q_base_stats["compactness"]),
            "q_base_overlap": float(q_base_stats["overlap"]),
            "q_enh_focus": float(q_enh_stats["focus"]),
            "q_enh_compactness": float(q_enh_stats["compactness"]),
            "q_enh_overlap": float(q_enh_stats["overlap"]),
        }

    if getattr(args, "response_calibration", False):
        prompt_map = text_conditioned_heatmap if text_conditioned_heatmap is not None else normalize_map(A_full)
        mask_enhanced_map = psme_heatmap if psme_heatmap is not None else normalize_map(A_full)
        candidate_mask = mg_mask if mg_mask is not None else M_fg
        selected_map = calibrate_response(
            base_heatmap=normalize_map(A_baseline),
            prompt_heatmap=prompt_map,
            mask_enhanced_heatmap=mask_enhanced_map,
            multiscale_heatmap=normalize_map(selected_map),
            candidate_mask=candidate_mask,
            alpha=getattr(args, "response_calibration_alpha", 0.15),
            positive_only=getattr(args, "response_calibration_positive_only", True),
        )

    A_full = ensure_finite_array("global_heatmap", A_full)
    A_tile = ensure_finite_array("tile_heatmap", A_tile)
    selected_map = ensure_finite_array("selected_map", selected_map)

    return {
        BASELINE_METHOD: {"map": A_baseline, "score": baseline_score},
        args.mg_method_name: {"map": A_full, "score": mg_score},
        MS_METHOD: {"map": selected_map, "score": selected_score},
        "debug": {
            "baseline_heatmap": A_baseline,
            "A_semantic": A_semantic,
            "A_local": A_local,
            "A_mask_seed": A_mask_seed,
            "mask_source_map": mask_source_map,
            "candidate_mask_source_map": candidate_mask_source_map,
            "candidate_mask_source_name": candidate_mask_source_name,
            "A_full": A_full,
            "A_tile": A_tile,
            "M_fg": M_fg,
            "tile_count": tile_count,
            "selected_map": selected_map,
            "mg_mask": mg_mask,
            "mg_mask_threshold": mg_mask_threshold,
            "text_score": text_score,
            "text_conditioned_heatmap": text_conditioned_heatmap,
            "text_mask_input": text_mask_input,
            "text_mask": text_mask,
            "psme_heatmap": psme_heatmap,
            "global_heatmap_norm": global_heatmap_norm,
            "local_weighted_heatmap": local_weighted_heatmap,
            "local_heatmap_norm": local_heatmap_norm,
            "final_heatmap": selected_map,
            "response_calibration": bool(getattr(args, "response_calibration", False)),
            "tile_guidance_scores_raw": tile_guidance_scores_raw,
            "tile_guidance_scores": tile_guidance_scores,
            "tile_weights": tile_weights,
            "reliability_stats": reliability_stats,
            "selection_stats": selection_stats,
            "iterative_refinement_trace": iterative_refinement_trace,
        },
    }


def subset_name(mask_ratio, is_anomaly):
    names = ["all"]
    if is_anomaly and mask_ratio < 0.01:
        names.append("small")
    if is_anomaly and mask_ratio < 0.005:
        names.append("tiny")
    return names


def resolve_eval_subsets(args):
    if getattr(args, "eval_all_subsets", False):
        return ["all", "small", "tiny"]
    raw = getattr(args, "eval_subsets", None)
    if raw is None or str(raw).strip() == "":
        return [args.subset]
    allowed = {"all", "small", "tiny"}
    values = []
    for part in str(raw).split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported eval subset: {item}")
        if item not in values:
            values.append(item)
    return values or [args.subset]


def should_keep_sample(sample, data_root, subset):
    if subset == "all" or not sample.is_anomaly:
        return True
    image = Image.open(sample.image_path)
    try:
        mask = load_mask(sample, image.size)
    finally:
        image.close()
    ratio = float(mask.sum() / max(mask.size, 1))
    sample_subsets = subset_name(ratio, True)
    return subset in sample_subsets


def should_keep_sample_for_inference(sample, data_root, args):
    if getattr(args, "anomaly_only_inference", False) and not sample.is_anomaly:
        return False
    if getattr(args, "eval_all_subsets", False) or getattr(args, "eval_subsets", None):
        return True
    return should_keep_sample(sample, data_root, args.subset)


def compute_subset_metrics(records, method, subset, pro_limit, metric_exact_max_pixels, metric_bins):
    if subset == "all":
        chosen = records
        anomalous_count = sum(1 for item in chosen if item["label"] == 1)
    else:
        anomalous = [item for item in records if item["label"] == 1 and subset in item["subsets"]]
        anomalous_count = len(anomalous)
        if anomalous_count == 0:
            return empty_metrics(0)
        normal = [item for item in records if item["label"] == 0]
        chosen = [*normal, *anomalous]

    maps = [item[method]["map"] for item in chosen]
    masks = [item["mask"] for item in chosen]
    pixel_scores = np.concatenate([item.reshape(-1) for item in maps]).astype(np.float32, copy=False)
    pixel_labels = np.concatenate([item.reshape(-1) for item in masks]).astype(np.uint8, copy=False)
    pixel_metrics = compute_pixel_metrics(
        pixel_scores,
        pixel_labels,
        exact_max_pixels=metric_exact_max_pixels,
        bins=metric_bins,
    )
    aupro_segauc_only = bool(getattr(sys.modules[__name__], "_AUPRO_SEGAUC_ONLY", False))
    seg_f1 = float("nan") if aupro_segauc_only else pixel_metrics["seg_f1"]
    pixel_ap = float("nan") if aupro_segauc_only else pixel_metrics["pixel_ap"]
    cls_auc = float("nan")
    cls_f1_value = float("nan")
    if not aupro_segauc_only:
        image_scores = [item[method]["score"] for item in chosen]
        image_labels = [item["label"] for item in chosen]
        cls_auc = compute_binary_auroc(image_scores, image_labels)
        cls_f1_value = f1_max(image_scores, image_labels)["f1_max"]
    return {
        "num_samples": int(anomalous_count if subset != "all" else len(chosen)),
        "au_pro": compute_legacy_au_pro(maps, masks, pro_limit, metric_exact_max_pixels, metric_bins),
        "seg_auc": pixel_metrics["seg_auc"],
        "seg_f1": seg_f1,
        "pixel_ap": pixel_ap,
        "cls_auc": cls_auc,
        "cls_f1": cls_f1_value,
    }


def compute_legacy_au_pro(maps, masks, pro_limit, metric_exact_max_pixels=DEFAULT_METRIC_EXACT_MAX_PIXELS, metric_bins=DEFAULT_METRIC_BINS):
    if not maps:
        return float("nan")
    if not any(np.any(mask > 0) for mask in masks):
        return float("nan")
    total_pixels = int(sum(np.asarray(item).size for item in maps))
    if total_pixels > int(metric_exact_max_pixels):
        print(
            f"[WARN] AU-PRO arrays contain {total_pixels} pixels; "
            f"using {metric_bins}-bin PRO approximation to avoid OOM."
        )
        return compute_pro_binned(maps, masks, pro_limit, metric_bins)
    try:
        pro_curve = legacy_compute_pro(anomaly_maps=maps, ground_truth_maps=masks)
        return float(legacy_trapezoid(pro_curve[0], pro_curve[1], x_max=pro_limit) / pro_limit)
    except (AssertionError, ValueError, RuntimeError):
        return pro_auc(maps, masks, pro_limit)


def compute_pro_binned(maps, masks, pro_limit, bins):
    bins = max(256, int(bins))
    score_min = min(float(np.nanmin(np.asarray(item, dtype=np.float32))) for item in maps)
    score_max = max(float(np.nanmax(np.asarray(item, dtype=np.float32))) for item in maps)
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_max - score_min < 1e-8:
        return float("nan")

    fp_hist = np.zeros(bins, dtype=np.float64)
    pro_hist = np.zeros(bins, dtype=np.float64)
    num_ok = 0
    num_regions = 0
    structure = np.ones((3, 3), dtype=int)
    scale = (bins - 1) / (score_max - score_min)

    for pred, mask in zip(maps, masks):
        pred = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.asarray(mask, dtype=np.uint8)
        scaled = np.floor((pred - score_min) * scale).astype(np.int32, copy=False)
        np.clip(scaled, 0, bins - 1, out=scaled)

        labeled, n_components = connected_components(mask > 0, structure)
        ok_mask = labeled == 0
        num_ok += int(np.sum(ok_mask))
        num_regions += int(n_components)
        if np.any(ok_mask):
            fp_hist += np.bincount(scaled[ok_mask].reshape(-1), minlength=bins)
        for component_idx in range(1, n_components + 1):
            region = labeled == component_idx
            region_size = int(np.sum(region))
            if region_size > 0:
                pro_hist += np.bincount(scaled[region].reshape(-1), minlength=bins) / float(region_size)

    if num_ok == 0 or num_regions == 0:
        return float("nan")

    fp = np.cumsum(fp_hist[::-1])
    pro = np.cumsum(pro_hist[::-1])
    fprs = np.r_[0.0, np.clip(fp / float(num_ok), 0.0, 1.0), 1.0]
    pros = np.r_[0.0, np.clip(pro / float(num_regions), 0.0, 1.0), 1.0]
    return float(legacy_trapezoid(fprs, pros, x_max=pro_limit) / pro_limit)


def empty_metrics(num_samples):
    return {
        "num_samples": int(num_samples),
        "au_pro": float("nan"),
        "seg_auc": float("nan"),
        "seg_f1": float("nan"),
        "pixel_ap": float("nan"),
        "cls_auc": float("nan"),
        "cls_f1": float("nan"),
    }


def fmt(value):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not np.isfinite(value) else f"{value:.6f}"


def normalize_map_for_visualization(array, low_percentile=70.0, high_percentile=99.2, gamma=0.85):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    values = array[finite]
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return normalize_map(array)
    norm = (array - low) / max(high - low, 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, gamma)
    norm[~finite] = 0.0
    return norm.astype(np.float32)


def heatmap_to_rgb_panel(array, size, low_percentile=70.0, high_percentile=99.2, gamma=0.85):
    norm = normalize_map_for_visualization(
        array,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        gamma=gamma,
    )
    heatmap_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb).resize(size, Image.BICUBIC)


def build_sparse_response_map(norm_map, activation_percentile=88.0):
    norm_map = np.asarray(norm_map, dtype=np.float32)
    finite = np.isfinite(norm_map)
    if not finite.any():
        return np.zeros_like(norm_map, dtype=np.float32)
    values = norm_map[finite]
    activation_percentile = min(max(float(activation_percentile), 0.0), 100.0)
    threshold = float(np.percentile(values, activation_percentile))
    if not np.isfinite(threshold):
        threshold = 0.0
    sparse = (norm_map - threshold) / max(1.0 - threshold, 1e-6)
    sparse = np.clip(sparse, 0.0, 1.0)
    sparse[~finite] = 0.0
    return sparse.astype(np.float32)


def sparse_heatmap_to_rgb_panel(
    array,
    size,
    low_percentile=70.0,
    high_percentile=99.2,
    gamma=0.85,
    activation_percentile=88.0,
):
    norm = normalize_map_for_visualization(
        array,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        gamma=gamma,
    )
    sparse = build_sparse_response_map(norm, activation_percentile=activation_percentile)
    heatmap_u8 = np.clip(sparse * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    heatmap_rgb *= sparse[..., None]
    return Image.fromarray(np.clip(heatmap_rgb, 0, 255).astype(np.uint8)).resize(size, Image.BICUBIC)


def binary_to_rgb_panel(array, size):
    binary = (np.asarray(array, dtype=np.float32) > 0).astype(np.uint8) * 255
    return Image.fromarray(binary).convert("RGB").resize(size, Image.NEAREST)


def overlay_heatmap_on_image(
    image,
    heatmap,
    alpha=0.38,
    low_percentile=70.0,
    high_percentile=99.2,
    gamma=0.85,
    activation_percentile=88.0,
):
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    norm = normalize_map_for_visualization(
        heatmap,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        gamma=gamma,
    )
    sparse = build_sparse_response_map(norm, activation_percentile=activation_percentile)
    heatmap_rgb = np.asarray(
        sparse_heatmap_to_rgb_panel(
            heatmap,
            image.size,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            gamma=gamma,
            activation_percentile=activation_percentile,
        ),
        dtype=np.float32,
    )
    alpha_map = (float(alpha) * sparse[..., None]).astype(np.float32)
    overlay = image_rgb * (1.0 - alpha_map) + heatmap_rgb * alpha_map
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def parse_vis_classes_arg(vis_classes):
    if vis_classes is None:
        return None
    items = [part.strip() for part in str(vis_classes).split(",")]
    values = {item for item in items if item}
    return values or None


def should_save_selected_visualization(args, class_name, sample, class_vis_count):
    if not getattr(args, "save_selected_visualizations_only", False):
        return False
    if not sample.is_anomaly and not getattr(args, "save_selected_include_good", False):
        return False
    allowed_classes = getattr(args, "_vis_classes_set", None)
    if allowed_classes is not None and class_name not in allowed_classes:
        return False
    max_vis = int(getattr(args, "max_vis_per_class", 0) or 0)
    if max_vis <= 0:
        return False
    return int(class_vis_count.get(class_name, 0)) < max_vis


def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_selected_visualization_bundle(output_dir, class_name, sample, image, mask, baseline_map, final_map, debug, args):
    sample_dir = (
        Path(output_dir)
        / "selected_visualizations"
        / class_name
        / sample.label_name
        / sample.image_path.stem
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_png(sample_dir / "original.png", image.convert("RGB"))
    save_png(sample_dir / "gt_mask.png", binary_to_rgb_panel(mask, image.size))
    save_png(
        sample_dir / "heatmap.png",
        sparse_heatmap_to_rgb_panel(
            final_map,
            image.size,
            low_percentile=float(getattr(args, "vis_percentile_low", 70.0)),
            high_percentile=float(getattr(args, "vis_percentile_high", 99.2)),
            gamma=float(getattr(args, "vis_gamma", 0.85)),
            activation_percentile=float(getattr(args, "vis_activation_percentile", 88.0)),
        ),
    )
    save_png(
        sample_dir / "overlay.png",
        overlay_heatmap_on_image(
            image,
            final_map,
            alpha=float(getattr(args, "vis_overlay_alpha", 0.38)),
            low_percentile=float(getattr(args, "vis_percentile_low", 70.0)),
            high_percentile=float(getattr(args, "vis_percentile_high", 99.2)),
            gamma=float(getattr(args, "vis_gamma", 0.85)),
            activation_percentile=float(getattr(args, "vis_activation_percentile", 88.0)),
        ),
    )

    if getattr(args, "save_npy_visualizations", False):
        np.save(sample_dir / "heatmap.npy", np.asarray(final_map, dtype=np.float32))

    local_layer = int(getattr(args, "local_feature_layer", 5))
    mask_source = str(getattr(args, "local_mask_source", "fused"))
    stage_to_value = {
        "A_semantic.png": debug.get("A_semantic"),
        f"A_local_l{local_layer}.png": debug.get("A_local"),
        f"A_mask_seed_{mask_source}.png": debug.get("A_mask_seed"),
        "A_full.png": debug.get("A_full"),
        "A_tile.png": debug.get("A_tile"),
        "A_psme.png": debug.get("psme_heatmap"),
        "candidate_mask.png": debug.get("mg_mask"),
        "A_final.png": debug.get("final_heatmap"),
        "baseline_heatmap.png": baseline_map,
    }
    for file_name, value in stage_to_value.items():
        if value is None:
            continue
        if file_name == "candidate_mask.png":
            save_png(
                sample_dir / file_name,
                heatmap_to_rgb_panel(value, image.size, low_percentile=0.0, high_percentile=100.0, gamma=1.0),
            )
        elif "mask" in file_name:
            save_png(sample_dir / file_name, binary_to_rgb_panel(value, image.size))
        else:
            save_png(
                sample_dir / file_name,
                sparse_heatmap_to_rgb_panel(
                    value,
                    image.size,
                    low_percentile=float(getattr(args, "vis_percentile_low", 70.0)),
                    high_percentile=float(getattr(args, "vis_percentile_high", 99.2)),
                    gamma=float(getattr(args, "vis_gamma", 0.85)),
                    activation_percentile=float(getattr(args, "vis_activation_percentile", 88.0)),
                ),
            )
    return sample_dir


def save_dual_layer_debug_bundle(output_dir, class_name, sample, image, final_map, debug, args):
    debug_root = getattr(args, "dual_layer_debug_dir", None)
    if debug_root is None:
        debug_root = Path(output_dir) / "dual_layer_debug"
    else:
        debug_root = Path(debug_root)
    sample_dir = debug_root / class_name / sample.image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_png(sample_dir / "input.png", image.convert("RGB"))
    if debug.get("A_semantic") is not None:
        save_png(
            sample_dir / "A_semantic.png",
            sparse_heatmap_to_rgb_panel(
                debug["A_semantic"],
                image.size,
                activation_percentile=float(getattr(args, "vis_activation_percentile", 88.0)),
            ),
        )
    if debug.get("A_local") is not None:
        save_png(
            sample_dir / f"A_local_l{int(getattr(args, 'local_feature_layer', 5))}.png",
            sparse_heatmap_to_rgb_panel(
                debug["A_local"],
                image.size,
                activation_percentile=float(getattr(args, "vis_activation_percentile", 88.0)),
            ),
        )
    if debug.get("A_mask_seed") is not None:
        save_png(
            sample_dir / f"A_mask_seed_{getattr(args, 'local_mask_source', 'fused')}.png",
            sparse_heatmap_to_rgb_panel(
                debug["A_mask_seed"],
                image.size,
                activation_percentile=float(getattr(args, "vis_activation_percentile", 88.0)),
            ),
        )
    if debug.get("mg_mask") is not None:
        save_png(sample_dir / "mg_mask.png", heatmap_to_rgb_panel(debug["mg_mask"], image.size))
    save_png(
        sample_dir / "A_final.png",
        sparse_heatmap_to_rgb_panel(
            final_map,
            image.size,
            activation_percentile=float(getattr(args, "vis_activation_percentile", 88.0)),
        ),
    )
    return sample_dir


def get_qualitative_method_labels(args):
    requested_prompt_style = str(getattr(args, "prompt_style", "default") or "default").strip().lower()
    if requested_prompt_style == "adaptive":
        return "Default-MG-MS", "LPS-SAPE-MG-MS"
    if requested_prompt_style == "spatial_aware":
        return "Default-MG-MS", "SAPE-MG-MS"
    return "Default-MG-MS", "Default-MG-MS"


def save_visualization(image, mask, baseline_map, mg_map, final_map, out_path, default_label, final_label, debug=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = image.size

    def heatmap_panel(array):
        norm = normalize_map(array)
        r = np.clip(2.0 * norm, 0, 1)
        g = np.clip(2.0 * (1.0 - np.abs(norm - 0.5)), 0, 1)
        b = np.clip(2.0 * (1.0 - norm), 0, 1)
        return Image.fromarray((np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)).resize(size, Image.BICUBIC)

    def binary_panel(array):
        binary = (np.asarray(array, dtype=np.float32) > 0).astype(np.uint8) * 255
        return Image.fromarray(binary).convert("RGB").resize(size, Image.NEAREST)

    def titled_panel(title, panel):
        title_height = 26
        canvas = Image.new("RGB", (panel.width, panel.height + title_height), color=(255, 255, 255))
        canvas.paste(panel, (0, title_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), str(title), fill=(0, 0, 0))
        return canvas

    panels = [
        ("Input Image", image.convert("RGB")),
        ("GT Mask", binary_panel(mask)),
        ("FG-CLIP baseline", heatmap_panel(baseline_map)),
        (default_label, heatmap_panel(mg_map)),
        (final_label, heatmap_panel(final_map)),
    ]
    if isinstance(debug, dict):
        for title, array, as_binary in [
            ("text_heatmap", debug.get("text_conditioned_heatmap"), False),
            ("text_mask", debug.get("text_mask"), False),
            ("psme", debug.get("psme_heatmap"), False),
            ("local_ms", debug.get("A_tile"), False),
            ("mg_mask", debug.get("mg_mask"), False),
        ]:
            if array is None:
                continue
            panels.append((title, binary_panel(array) if as_binary else heatmap_panel(array)))

    titled_panels = [titled_panel(title, panel) for title, panel in panels]
    canvas = Image.new("RGB", (size[0] * len(titled_panels), titled_panels[0].height))
    for idx, panel in enumerate(titled_panels):
        canvas.paste(panel, (idx * size[0], 0))
    canvas.save(out_path)


def save_heatmap_tiff(path, heatmap):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = ensure_finite_array(str(path), np.asarray(heatmap, dtype=np.float32))
    ok, encoded = cv2.imencode(".tiff", array)
    if not ok:
        raise RuntimeError(f"Failed to encode heatmap: {path}")
    encoded.tofile(str(path))


def save_method_heatmaps(output_dir, dataset, class_name, sample, outputs, method_names):
    for method in method_names:
        out_path = (
            output_dir
            / "heatmaps"
            / method
            / dataset
            / class_name
            / sample.split
            / sample.label_name
            / f"{sample.image_path.stem}.tiff"
        )
        save_heatmap_tiff(out_path, outputs[method]["map"])


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calibrate_ncrs_tau_risk(good_paths, baseline_ctx, mg_ctx, baseline_stats, mg_stats, args):
    calibration_args = SimpleNamespace(**vars(args))
    calibration_args.normal_calibrated_selection = False
    calibration_args.safe_response_selection = False
    calibration_args.save_intermediate = False
    calibration_args.save_visualizations = False
    calibration_args.save_heatmaps = False

    enhanced_heatmaps = []
    candidate_masks = []
    for image_path in good_paths:
        with Image.open(image_path).convert("RGB") as image:
            outputs = infer_ms_fb_mg(
                image,
                baseline_ctx,
                mg_ctx,
                baseline_stats,
                mg_stats,
                calibration_args,
                ncrs_tau_risk=None,
            )
        enhanced_heatmaps.append(outputs[MS_METHOD]["map"])
        candidate_masks.append(outputs["debug"].get("mg_mask"))

    return calibrate_risk_threshold(
        enhanced_heatmaps,
        candidate_masks=candidate_masks,
        risk_quantile=getattr(args, "pars_risk_quantile", 0.95),
        area_q=getattr(args, "pars_area_q", 0.80),
    )


def main():
    args = parse_args()
    global _AUPRO_SEGAUC_ONLY
    _AUPRO_SEGAUC_ONLY = bool(getattr(args, "aupro_segauc_only", False))
    resolve_ablation_flags(args)
    set_progress_style(args.progress_style)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[INFO] method = {args.method}")
        print(f"[INFO] method_name = {args.resolved_method_name}")
        print(f"[INFO] enable_prompt_ensemble = {bool(args.prompt_ensemble)}")
        print(f"[INFO] prompt_style = {args.prompt_style}")
        print(f"[INFO] enable_candidate_mask = {bool(args.enable_positive_fusion)}")
        print(f"[INFO] enable_multiscale = {bool(args.enable_ms)}")
        print(f"[INFO] enable_response_calibration = {bool(args.response_calibration)}")
        print(f"[INFO] enable_reliability_ms_fusion = {bool(args.reliability_ms_fusion)}")
        print(f"[INFO] enable_safe_response_selection = {bool(args.safe_response_selection)}")
        print(f"[INFO] enable_text_conditioned_mask = {bool(args.text_conditioned_mask)}")
        print(f"[INFO] enable_normal_calibrated_selection = {bool(args.normal_calibrated_selection)}")
        print(f"[INFO] enable_fg = {bool(args.enable_fg)}")
        print(f"[INFO] foreground_mode = {args.foreground_mode}")
        print(f"[INFO] global_guided_fusion = {bool(args.global_guided_fusion)}")
        print(f"[INFO] reliability_beta = {float(args.reliability_beta):.4f}")
        print(f"[INFO] pars_margin = {float(args.pars_margin):.4f}")
        print(f"[INFO] pars_risk_quantile = {float(args.pars_risk_quantile):.4f}")
        print(f"[INFO] pars_topk_ratio = {float(args.pars_topk_ratio):.4f}")
        print(f"[INFO] pars_area_q = {float(args.pars_area_q):.4f}")
        print(f"[INFO] pars_save_selection_stats = {bool(args.pars_save_selection_stats)}")
        print(f"[INFO] pars_enhanced_bias = {float(args.pars_enhanced_bias):.4f}")
        return

    if args.dataset is None or args.data_root is None:
        raise ValueError("--dataset and --data_root are required unless --dry_run is used.")

    data_root = args.data_root.resolve()
    visa_split = args.visa_split or data_root / "split_csv" / "1cls.csv"
    adaptive_policy_path = args.adaptive_prompt_policy_path
    if adaptive_policy_path is None:
        adaptive_policy_path = REPO_ROOT / "configs" / "adaptive_prompt_policy.json"
    adaptive_policy = load_adaptive_prompt_policy(adaptive_policy_path)

    args.mg_refiner_dir, args.mg_method_name = resolve_mg_refiner_runtime(args)
    target_method = args.resolved_method_name
    if isinstance(target_method, str) and "," in target_method:
        method_names = [item.strip() for item in target_method.split(",") if item.strip()]
        target_method = method_names[0]
    else:
        method_names = [target_method]

    print(f"[INFO] method          = {args.method}")
    print(f"[INFO] method_name     = {target_method}")
    print(f"[INFO] mg_method_name  = {args.mg_method_name}")
    print(f"[INFO] dataset         = {args.dataset}")
    print(f"[INFO] data_root       = {data_root}")
    print(f"[INFO] model_path      = {args.model_path}")
    print(f"[INFO] mg_model_path   = {args.mg_model_path}")
    print(f"[INFO] output_dir      = {output_dir}")
    print(f"[INFO] mg_refiner_dir  = {args.mg_refiner_dir}")
    print(f"[INFO] subset          = {args.subset}")
    print(f"[INFO] enable_mg       = {args.enable_mg}")
    print(f"[INFO] enable_fg       = {args.enable_fg}")
    print(f"[INFO] enable_ms       = {args.enable_ms}")
    print(f"[INFO] positive_fusion = {args.enable_positive_fusion}")
    print(f"[INFO] enable_prompt_ensemble = {bool(args.prompt_ensemble)}")
    print(f"[INFO] enable_candidate_mask = {bool(args.enable_positive_fusion)}")
    print(f"[INFO] enable_multiscale = {bool(args.enable_ms)}")
    print(f"[INFO] enable_response_calibration = {bool(args.response_calibration)}")
    print(f"[INFO] enable_reliability_ms_fusion = {bool(args.reliability_ms_fusion)}")
    print(f"[INFO] enable_safe_response_selection = {bool(args.safe_response_selection)}")
    print(f"[INFO] enable_text_conditioned_mask = {bool(args.text_conditioned_mask)}")
    print(f"[INFO] enable_normal_calibrated_selection = {bool(args.normal_calibrated_selection)}")
    print(f"[INFO] mg_mask_type    = {args.mg_mask_type}")
    print(f"[INFO] soft_mask_gamma = {args.soft_mask_gamma}")
    print(f"[INFO] prompt_ensemble = {args.prompt_ensemble}")
    print(f"[INFO] mg_start_layer = {args.mg_start_layer}")
    print(f"[INFO] mg_end_layer = {args.mg_end_layer}")
    print(f"[INFO] attention_bias_eta = {args.attention_bias_eta}")
    print(f"[INFO] prompt_style = {args.prompt_style}")
    print(f"[INFO] adaptive_prompt_policy_path = {adaptive_policy_path}")
    print(f"[INFO] global_guided_fusion = {args.global_guided_fusion}")
    print(f"[INFO] tile_weight_temperature = {args.tile_weight_temperature}")
    print(f"[INFO] tile_topk_ratio = {args.tile_topk_ratio}")
    print(f"[INFO] alpha = {args.alpha}")
    print(f"[INFO] reliability_beta = {args.reliability_beta}")
    print(f"[INFO] pars_margin = {args.pars_margin}")
    print(f"[INFO] pars_risk_quantile = {args.pars_risk_quantile}")
    print(f"[INFO] pars_topk_ratio = {args.pars_topk_ratio}")
    print(f"[INFO] pars_area_q = {args.pars_area_q}")
    print(f"[INFO] pars_save_selection_stats = {args.pars_save_selection_stats}")
    print(f"[INFO] pars_enhanced_bias = {args.pars_enhanced_bias}")
    print(f"[INFO] save_intermediate = {args.save_intermediate}")
    print(f"[INFO] save_selected_visualizations_only = {args.save_selected_visualizations_only}")
    print(f"[INFO] vis_classes = {args.vis_classes}")
    print(f"[INFO] max_vis_per_class = {args.max_vis_per_class}")
    print(f"[INFO] save_npy_visualizations = {args.save_npy_visualizations}")
    print(f"[INFO] subset = {args.subset}")
    print(f"[INFO] eval_subsets = {resolve_eval_subsets(args)}")
    print(f"[INFO] dual_layer_guidance = {bool(args.dual_layer_guidance)}")
    print(f"[INFO] local_feature_layer = {int(args.local_feature_layer)}")
    print(f"[INFO] local_mask_source = {args.local_mask_source}")
    print(f"[INFO] semantic_mask_weight = {float(args.semantic_mask_weight):.4f}")
    print(f"[INFO] local_mask_weight = {float(args.local_mask_weight):.4f}")
    print(f"[INFO] save_dual_layer_debug = {bool(args.save_dual_layer_debug)}")

    baseline_model, baseline_tokenizer, baseline_processor = load_model(args.model_path.resolve())
    if args.enable_mg:
        mg_model, mg_tokenizer, mg_processor = load_model(args.mg_model_path.resolve())
    else:
        mg_model, mg_tokenizer, mg_processor = baseline_model, baseline_tokenizer, baseline_processor

    baseline_args = make_runtime_args(args, use_mg=False, feature_layer=0)
    mg_args = make_runtime_args(
        args,
        use_mg=args.enable_mg,
        mg_refiner_dir=args.mg_refiner_dir,
        mg_refiner_checkpoint=args.mg_refiner_checkpoint,
        feature_layer=0,
    )
    local_args = make_runtime_args(args, use_mg=False, feature_layer=args.local_feature_layer)
    attach_mg_refiner(mg_args)

    classes = args.classes or discover_classes(args.dataset, data_root, visa_split if args.dataset == "visa" else None)
    args._vis_classes_set = parse_vis_classes_arg(args.vis_classes)
    eval_subsets = resolve_eval_subsets(args)
    method_rows = []
    all_rows = []
    pars_selection_rows = []
    pars_summary_rows = []
    raw_json = {"config": vars(args), "classes": {}}
    selected_vis_count_by_class = {}

    for class_name in classes:
        print(f"\n[INFO] Processing {args.dataset}/{class_name}")
        train_samples, test_samples = load_dataset_samples(args.dataset, data_root, class_name, visa_split)
        test_samples = [sample for sample in test_samples if should_keep_sample_for_inference(sample, data_root, args)]
        if args.max_test_images:
            test_samples = test_samples[: args.max_test_images]
        if not test_samples:
            print(f"[WARN] No test samples matched subset={args.subset} for {class_name}, skipping.")
            continue
        good_dir = get_good_dir(args.dataset, data_root, class_name, train_samples)
        activate_object_mg_refiner(mg_args, class_name)
        base_good_prompts, base_defect_prompts = get_text_banks(
            args.dataset,
            class_name,
            baseline_tokenizer,
            baseline_model,
        )
        prompt_resolution = resolve_prompt_style_for_class(
            args.dataset,
            class_name,
            args.prompt_style,
            adaptive_policy=adaptive_policy,
        )
        effective_prompt_style = prompt_resolution["selected_prompt_style"]
        prompt_bundle = build_prompts(
            class_name=class_name,
            prompt_style=effective_prompt_style,
            base_normal_prompts=base_good_prompts,
            base_abnormal_prompts=base_defect_prompts,
            enable_prompt_ensemble=bool(args.prompt_ensemble),
        )
        prompt_bundle["requested_prompt_style"] = prompt_resolution["requested_prompt_style"]
        prompt_bundle["adaptive_selected_prompt_style"] = prompt_resolution["selected_prompt_style"]
        prompt_bundle["adaptive_policy_name"] = prompt_resolution["adaptive_policy_name"]
        if args.prompt_style == "adaptive":
            print(f"[INFO] prompt_style = adaptive")
            print(f"[INFO] adaptive selected prompt_style = {prompt_resolution['selected_prompt_style']}")
            print(f"[INFO] adaptive policy = {prompt_resolution['adaptive_policy_name'] or 'default_fallback'}")
            print(f"[INFO] class_name = {class_name}")
        else:
            print(f"[INFO] prompt_style = {prompt_bundle['prompt_style']}")
        print(f"[INFO] num_normal_prompts = {len(prompt_bundle['normal_prompts'])}")
        print(f"[INFO] num_general_abnormal_prompts = {len(prompt_bundle['general_abnormal_prompts'])}")
        print(f"[INFO] num_spatial_abnormal_prompts = {len(prompt_bundle['spatial_abnormal_prompts'])}")
        print(f"[INFO] num_defect_type_prompts = {len(prompt_bundle['defect_type_prompts'])}")
        print(f"[INFO] num_abnormal_prompts = {len(prompt_bundle['abnormal_prompts'])}")
        prompt_stats_path = save_prompt_stats(output_dir, class_name, prompt_bundle)
        print(f"[INFO] prompt_stats = {prompt_stats_path}")

        baseline_text_features = build_text_feature_bundle(baseline_tokenizer, baseline_model, prompt_bundle)
        print(f"[INFO] building baseline prototype for {class_name}")
        baseline_proto, baseline_memory, baseline_good_paths = build_good_prototype(good_dir, baseline_processor, baseline_model, baseline_args)
        print(f"[INFO] calibrating baseline threshold for {class_name}")
        _, _, _, _, baseline_raw_mean, baseline_raw_std = calibrate_threshold(
            baseline_good_paths,
            baseline_processor,
            baseline_model,
            baseline_proto,
            baseline_memory,
            baseline_text_features.scoring_good_text,
            baseline_text_features.scoring_defect_text,
            baseline_args,
        )
        if args.enable_mg:
            mg_text_features = build_text_feature_bundle(mg_tokenizer, mg_model, prompt_bundle)
            print(f"[INFO] building mg prototype for {class_name}")
            mg_proto, mg_memory, mg_good_paths = build_good_prototype(good_dir, mg_processor, mg_model, mg_args)
            print(f"[INFO] calibrating mg threshold for {class_name}")
            _, _, _, _, mg_raw_mean, mg_raw_std = calibrate_threshold(
                mg_good_paths,
                mg_processor,
                mg_model,
                mg_proto,
                mg_memory,
                mg_text_features.scoring_good_text,
                mg_text_features.scoring_defect_text,
                mg_args,
            )
            mg_stats = (mg_raw_mean, mg_raw_std)
            mg_ctx = build_inference_context(mg_processor, mg_model, mg_proto, mg_memory, mg_text_features, mg_args)
        else:
            mg_stats = (baseline_raw_mean, baseline_raw_std)
            mg_ctx = build_inference_context(
                baseline_processor,
                baseline_model,
                baseline_proto,
                baseline_memory,
                baseline_text_features,
                baseline_args,
            )
        baseline_stats = (baseline_raw_mean, baseline_raw_std)
        baseline_ctx = build_inference_context(
            baseline_processor,
            baseline_model,
            baseline_proto,
            baseline_memory,
            baseline_text_features,
            baseline_args,
        )
        local_ctx = None
        local_stats = None
        if args.dual_layer_guidance:
            try:
                print(f"[INFO] building local layer-{int(args.local_feature_layer)} prototype for {class_name}")
                local_proto, local_memory, local_good_paths = build_good_prototype(good_dir, baseline_processor, baseline_model, local_args)
                print(f"[INFO] calibrating local layer-{int(args.local_feature_layer)} threshold for {class_name}")
                _, _, _, _, local_raw_mean, local_raw_std = calibrate_threshold(
                    local_good_paths,
                    baseline_processor,
                    baseline_model,
                    local_proto,
                    local_memory,
                    baseline_text_features.scoring_good_text,
                    baseline_text_features.scoring_defect_text,
                    local_args,
                )
                local_stats = (local_raw_mean, local_raw_std)
                local_ctx = build_inference_context(
                    baseline_processor,
                    baseline_model,
                    local_proto,
                    local_memory,
                    baseline_text_features,
                    local_args,
                )
            except Exception as exc:
                print(f"[WARN] failed to build local dual-layer context for {class_name}; semantic fallback will be used. Reason: {exc}")
                local_ctx = None
                local_stats = None
        ncrs_tau_risk = None
        if args.normal_calibrated_selection:
            calibration_paths = mg_good_paths if args.enable_mg else baseline_good_paths
            print(f"[INFO] calibrating NCRS risk threshold for {class_name}")
            ncrs_tau_risk = calibrate_ncrs_tau_risk(
                calibration_paths,
                baseline_ctx,
                mg_ctx,
                baseline_stats,
                mg_stats,
                args,
            )
            print(f"[INFO] NCRS tau_risk = {float(ncrs_tau_risk):.6f}")

        records = []
        class_pars_rows = []
        tiny_vis_count = 0
        small_vis_count = 0
        first_debug = True
        sample_iter = iter_progress(test_samples, desc=f"infer {class_name}", total=len(test_samples))
        for sample in sample_iter:
            image = Image.open(sample.image_path).convert("RGB")
            mask = load_mask(sample, image.size)
            mask_ratio = float(mask.sum() / max(mask.size, 1))
            outputs = infer_ms_fb_mg(
                image,
                baseline_ctx,
                mg_ctx,
                baseline_stats,
                mg_stats,
                args,
                ncrs_tau_risk=ncrs_tau_risk,
                local_ctx=local_ctx,
                local_stats=local_stats,
            )
            sample_subsets = subset_name(mask_ratio, sample.is_anomaly)
            if not args.visualization_only:
                record = {
                    "sample": sample,
                    "mask": mask,
                    "label": int(sample.is_anomaly),
                    "mask_ratio": mask_ratio,
                    "subsets": sample_subsets,
                }
                for method_name in method_names:
                    source = outputs.get(method_name)
                    if source is None:
                        source = outputs[MS_METHOD]
                    record[method_name] = {
                        "map": source["map"],
                        "score": source["score"],
                    }
                records.append(record)

            debug = outputs["debug"]
            if debug["selection_stats"] is not None:
                class_pars_rows.append(
                    {
                        "dataset": args.dataset,
                        "class_name": class_name,
                        "image_id": sample.image_path.stem,
                        "q_base": float(debug["selection_stats"]["q_base"]),
                        "q_enh": float(debug["selection_stats"]["q_enh"]),
                        "r_enh": float(debug["selection_stats"].get("r_enh", float("nan"))),
                        "tau_risk": float(debug["selection_stats"].get("tau_risk", float("nan"))),
                        "selected_source": str(debug["selection_stats"]["selected_source"]),
                        "image_score_raw": float(debug["selection_stats"].get("image_score_raw", outputs[MS_METHOD]["score"])),
                    }
                )
            if first_debug:
                print(f"[INFO] tile_count      = {debug['tile_count']}")
                print_stats("A_full", outputs[args.mg_method_name]["map"])
                if debug.get("A_semantic") is not None:
                    print_stats("A_semantic", debug["A_semantic"])
                if debug.get("A_local") is not None:
                    print_stats("A_local", debug["A_local"])
                if debug.get("A_mask_seed") is not None:
                    print_stats("A_mask_seed", debug["A_mask_seed"])
                print_stats("A_tile", debug["A_tile"])
                print_stats("A_final", debug["selected_map"])
                if args.enable_fg:
                    print(f"[INFO] foreground mean = {debug['M_fg'].mean():.6f}")
                else:
                    print(f"[INFO] default_mask mean = {debug['M_fg'].mean():.6f}")
                if debug["text_conditioned_heatmap"] is not None:
                    print_stats("text_conditioned_heatmap", debug["text_conditioned_heatmap"])
                    print_stats("text_mask", debug["text_mask"])
                    print_stats("psme_heatmap", debug["psme_heatmap"])
                if debug["tile_guidance_scores"] is not None:
                    print(
                        f"[INFO] tile guidance scores raw = "
                        f"{np.array2string(np.asarray(debug['tile_guidance_scores_raw']), precision=4)}"
                    )
                    print(
                        f"[INFO] tile guidance scores normalized = "
                        f"{np.array2string(np.asarray(debug['tile_guidance_scores']), precision=4)}"
                    )
                    print(
                        f"[INFO] tile weights = "
                        f"{np.array2string(np.asarray(debug['tile_weights']), precision=4)}"
                    )
                if debug["global_heatmap_norm"] is not None:
                    print_stats("H_global_norm", debug["global_heatmap_norm"])
                if debug["local_heatmap_norm"] is not None:
                    print_stats("H_local_norm", debug["local_heatmap_norm"])
                if debug["final_heatmap"] is not None and args.global_guided_fusion:
                    print_stats("H_final", debug["final_heatmap"])
                if debug["mg_mask"] is not None:
                    print(f"[INFO] mg_mask_type = {args.mg_mask_type}")
                    print(f"[INFO] soft_mask_gamma = {args.soft_mask_gamma}")
                    print(f"[INFO] mask threshold = {float(debug['mg_mask_threshold']):.4f}")
                    print(
                        f"[INFO] mask min={float(np.min(debug['mg_mask'])):.4f} "
                        f"max={float(np.max(debug['mg_mask'])):.4f} "
                        f"mean={float(np.mean(debug['mg_mask'])):.4f}"
                    )
                if debug["reliability_stats"] is not None:
                    stats = debug["reliability_stats"]
                    print(f"[INFO] reliability_weight = {float(stats['weight']):.6f}")
                    print(f"[INFO] R_focus = {float(stats['R_focus']):.6f}")
                    print(f"[INFO] R_spread = {float(stats['R_spread']):.6f}")
                    print(f"[INFO] R_overlap = {float(stats['R_overlap']):.6f}")
                if debug["selection_stats"] is not None:
                    stats = debug["selection_stats"]
                    print(f"[INFO] q_base = {float(stats['q_base']):.6f}")
                    print(f"[INFO] q_enh = {float(stats['q_enh']):.6f}")
                    if "r_enh" in stats:
                        print(f"[INFO] r_enh = {float(stats['r_enh']):.6f}")
                    if "tau_risk" in stats:
                        print(f"[INFO] tau_risk = {float(stats['tau_risk']):.6f}")
                    print(f"[INFO] selected_source = {stats['selected_source']}")
                first_debug = False

            if args.save_heatmaps:
                heatmap_outputs = {
                    BASELINE_METHOD: outputs[BASELINE_METHOD],
                }
                for method_name in method_names:
                    source = outputs.get(method_name)
                    if source is None:
                        source = outputs[MS_METHOD]
                    heatmap_outputs[method_name] = {"map": source["map"], "score": source["score"]}
                save_method_heatmaps(
                    output_dir,
                    args.dataset,
                    class_name,
                    sample,
                    heatmap_outputs,
                    [BASELINE_METHOD, target_method],
                )

            if args.save_intermediate:
                save_intermediate_outputs(output_dir, image, args.dataset, class_name, sample, outputs["debug"])

            if args.save_dual_layer_debug and args.dual_layer_guidance:
                saved_dual_dir = save_dual_layer_debug_bundle(
                    output_dir=output_dir,
                    class_name=class_name,
                    sample=sample,
                    image=image,
                    final_map=outputs[MS_METHOD]["map"],
                    debug=outputs["debug"],
                    args=args,
                )
                if first_debug:
                    print(f"[INFO] dual_layer_debug_dir = {saved_dual_dir.parent}")

            if should_save_selected_visualization(args, class_name, sample, selected_vis_count_by_class):
                saved_dir = save_selected_visualization_bundle(
                    output_dir=output_dir,
                    class_name=class_name,
                    sample=sample,
                    image=image,
                    mask=mask,
                    baseline_map=outputs[BASELINE_METHOD]["map"],
                    final_map=outputs[MS_METHOD]["map"],
                    debug=outputs["debug"],
                    args=args,
                )
                selected_vis_count_by_class[class_name] = int(selected_vis_count_by_class.get(class_name, 0)) + 1
                print(f"[INFO] selected visualization saved = {saved_dir}")
                if args.visualization_only:
                    max_vis = int(getattr(args, "max_vis_per_class", 0) or 0)
                    if max_vis > 0 and int(selected_vis_count_by_class.get(class_name, 0)) >= max_vis:
                        print(f"[INFO] visualization_only reached max_vis_per_class={max_vis} for {class_name}, stopping this class early.")
                        break

            if (
                args.save_visualizations
                and not args.save_selected_visualizations_only
                and sample.is_anomaly
                and ("tiny" in sample_subsets or "small" in sample_subsets)
            ):
                can_save = ("tiny" in sample_subsets and tiny_vis_count < 5) or ("small" in sample_subsets and small_vis_count < 5)
                if can_save:
                    out_name = f"{sample.label_name}_{sample.image_path.stem}.png"
                    default_label, final_label = get_qualitative_method_labels(args)
                    save_visualization(
                        image=image,
                        mask=mask,
                        baseline_map=outputs[BASELINE_METHOD]["map"],
                        mg_map=outputs[args.mg_method_name]["map"],
                        final_map=outputs[MS_METHOD]["map"],
                        default_label=default_label,
                        final_label=final_label,
                        out_path=output_dir / "vis" / args.dataset / class_name / out_name,
                        debug=outputs["debug"],
                    )
                    tiny_vis_count += int("tiny" in sample_subsets)
                    small_vis_count += int("small" in sample_subsets)
        if class_pars_rows:
            pars_selection_rows.extend(class_pars_rows)
            q_base_values = np.asarray([row["q_base"] for row in class_pars_rows], dtype=np.float32)
            q_enh_values = np.asarray([row["q_enh"] for row in class_pars_rows], dtype=np.float32)
            r_enh_values = np.asarray([row["r_enh"] for row in class_pars_rows], dtype=np.float32)
            baseline_count = sum(1 for row in class_pars_rows if row["selected_source"] == "baseline")
            enhanced_count = sum(1 for row in class_pars_rows if row["selected_source"] == "enhanced")
            total_count = len(class_pars_rows)
            class_summary = {
                "class_name": class_name,
                "num_samples": total_count,
                "selected_baseline_count": baseline_count,
                "selected_enhanced_count": enhanced_count,
                "selected_baseline_ratio": baseline_count / max(total_count, 1),
                "selected_enhanced_ratio": enhanced_count / max(total_count, 1),
                "mean_q_base": float(np.mean(q_base_values)) if q_base_values.size else float("nan"),
                "mean_q_enh": float(np.mean(q_enh_values)) if q_enh_values.size else float("nan"),
                "std_q_base": float(np.std(q_base_values)) if q_base_values.size else float("nan"),
                "std_q_enh": float(np.std(q_enh_values)) if q_enh_values.size else float("nan"),
                "mean_r_enh": float(np.mean(r_enh_values)) if r_enh_values.size else float("nan"),
                "std_r_enh": float(np.std(r_enh_values)) if r_enh_values.size else float("nan"),
                "tau_risk": float(ncrs_tau_risk) if ncrs_tau_risk is not None else float("nan"),
            }
            pars_summary_rows.append({"dataset": args.dataset, **class_summary})
            if args.pars_save_selection_stats:
                class_stats_dir = output_dir / "selection_stats" / args.dataset / class_name
                summary_fields = [
                    "class_name",
                    "num_samples",
                    "selected_baseline_count",
                    "selected_enhanced_count",
                    "selected_baseline_ratio",
                    "selected_enhanced_ratio",
                    "mean_q_base",
                    "mean_q_enh",
                    "std_q_base",
                    "std_q_enh",
                    "mean_r_enh",
                    "std_r_enh",
                    "tau_risk",
                ]
                write_csv(class_stats_dir / "pars_selection_records.csv", class_pars_rows, [
                    "image_id",
                    "class_name",
                    "q_base",
                    "q_enh",
                    "r_enh",
                    "tau_risk",
                    "selected_source",
                    "image_score_raw",
                ])
                write_csv(class_stats_dir / "pars_selection_summary.csv", [class_summary], summary_fields)
                with open(class_stats_dir / "pars_selection_summary.json", "w", encoding="utf-8") as f:
                    json.dump(class_summary, f, indent=2, ensure_ascii=False)
        raw_json["classes"][class_name] = {"num_test": len(records) if not args.visualization_only else len(test_samples)}
        if args.visualization_only:
            continue
        for method in method_names:
            for subset in eval_subsets:
                metrics = compute_subset_metrics(
                    records,
                    method,
                    subset,
                    args.pro_limit,
                    args.metric_exact_max_pixels,
                    args.metric_bins,
                )
                row = {
                    "method": method,
                    "dataset": args.dataset,
                    "class": class_name,
                    "subset": subset,
                    "num_samples": metrics["num_samples"],
                    "au_pro": fmt(metrics["au_pro"]),
                    "seg_auc": fmt(metrics["seg_auc"]),
                    "seg_f1": fmt(metrics["seg_f1"]),
                    "pixel_ap": fmt(metrics["pixel_ap"]),
                    "cls_auc": fmt(metrics["cls_auc"]),
                    "cls_f1": fmt(metrics["cls_f1"]),
                }
                method_rows.append(row)
                print(
                    f"[RESULT] {method} {args.dataset}/{class_name}/{subset}: "
                    f"AU-PRO={row['au_pro']} SegAUC={row['seg_auc']} SegF1={row['seg_f1']} "
                    f"PixelAP={row['pixel_ap']} ClsAUC={row['cls_auc']}"
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.visualization_only:
        with open(output_dir / "metrics_raw.json", "w", encoding="utf-8") as f:
            json.dump(raw_json, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[RESULT] selected_visualizations = {output_dir / 'selected_visualizations'}")
        print(f"[RESULT] prompt_stats             = {output_dir / 'prompt_stats'}")
        print(f"[RESULT] metrics_raw             = {output_dir / 'metrics_raw.json'}")
        return

    fields = ["method", "dataset", "class", "subset", "num_samples", "au_pro", "seg_auc", "seg_f1", "pixel_ap", "cls_auc", "cls_f1"]
    write_csv(output_dir / "metrics_per_class.csv", method_rows, fields)
    write_csv(output_dir / "metrics_small_subset.csv", method_rows, fields)
    if pars_selection_rows and args.pars_save_selection_stats:
        write_csv(
            output_dir / "pars_selection_all.csv",
            pars_selection_rows,
            [
                "dataset",
                "class_name",
                "image_id",
                "q_base",
                "q_enh",
                "r_enh",
                "tau_risk",
                "selected_source",
                "image_score_raw",
            ],
        )
    if pars_summary_rows and args.pars_save_selection_stats:
        write_csv(
            output_dir / "pars_selection_summary.csv",
            pars_summary_rows,
            [
                "dataset",
                "class_name",
                "num_samples",
                "selected_baseline_count",
                "selected_enhanced_count",
                "selected_baseline_ratio",
                "selected_enhanced_ratio",
                "mean_q_base",
                "mean_q_enh",
                "std_q_base",
                "std_q_enh",
                "mean_r_enh",
                "std_r_enh",
                "tau_risk",
            ],
        )
        with open(output_dir / "pars_selection_summary.json", "w", encoding="utf-8") as f:
            json.dump(pars_summary_rows, f, indent=2, ensure_ascii=False)
    summary_rows = write_summary(output_dir / "summary_mean.csv", method_rows, args.dataset, method_names)
    write_csv(
        output_dir / "metrics_all.csv",
        summary_rows,
        ["method", "dataset", "subset", "mean_au_pro", "mean_seg_auc", "mean_seg_f1", "mean_pixel_ap", "mean_cls_auc", "mean_cls_f1"],
    )
    metrics_json = {}
    summary_subset = "all" if "all" in eval_subsets else eval_subsets[0]
    summary_candidates = [row for row in summary_rows if row["method"] == target_method and row["subset"] == summary_subset]
    if summary_candidates:
        summary_row = summary_candidates[0]
        metrics_json = {
            "dataset": args.dataset,
            "method": target_method,
            "subset": summary_subset,
            "mean_au_pro": safe_float(summary_row["mean_au_pro"]),
            "mean_segmentation_au_roc": safe_float(summary_row["mean_seg_auc"]),
            "mean_segmentation_f1_max": safe_float(summary_row["mean_seg_f1"]),
            "mean_pixel_ap": safe_float(summary_row["mean_pixel_ap"]),
            "mean_classification_au_roc": safe_float(summary_row["mean_cls_auc"]),
            "mean_classification_f1_max": safe_float(summary_row["mean_cls_f1"]),
        }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    with open(output_dir / "metrics_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[RESULT] metrics_all          = {output_dir / 'metrics_all.csv'}")
    print(f"[RESULT] metrics_per_class    = {output_dir / 'metrics_per_class.csv'}")
    print(f"[RESULT] metrics_small_subset = {output_dir / 'metrics_small_subset.csv'}")
    print(f"[RESULT] summary_mean         = {output_dir / 'summary_mean.csv'}")
    if args.pars_save_selection_stats and pars_selection_rows:
        print(f"[RESULT] pars_selection_all   = {output_dir / 'pars_selection_all.csv'}")
        print(f"[RESULT] pars_selection_summary = {output_dir / 'pars_selection_summary.csv'}")
    if args.save_heatmaps:
        print(f"[RESULT] heatmaps             = {output_dir / 'heatmaps'}")
    if args.save_visualizations:
        print(f"[RESULT] visualizations       = {output_dir / 'vis'}")
    if args.save_selected_visualizations_only:
        print(f"[RESULT] selected_visualizations = {output_dir / 'selected_visualizations'}")


def print_stats(name, array):
    array = np.asarray(array, dtype=np.float32)
    print(f"[INFO] {name:<14} min={array.min():.6f} max={array.max():.6f} mean={array.mean():.6f}")


def write_summary(path, rows, dataset, method_names):
    summary_rows = []
    for method in method_names:
        for subset in sorted({row["subset"] for row in rows}):
            selected = [row for row in rows if row["method"] == method and row["subset"] == subset]
            out = {"method": method, "dataset": dataset, "subset": subset}
            for src, dst in [
                ("au_pro", "mean_au_pro"),
                ("seg_auc", "mean_seg_auc"),
                ("seg_f1", "mean_seg_f1"),
                ("pixel_ap", "mean_pixel_ap"),
                ("cls_auc", "mean_cls_auc"),
                ("cls_f1", "mean_cls_f1"),
            ]:
                values = [float(row[src]) for row in selected if row[src] != "nan"]
                out[dst] = fmt(np.mean(values) if values else float("nan"))
            summary_rows.append(out)
    write_csv(
        path,
        summary_rows,
        ["method", "dataset", "subset", "mean_au_pro", "mean_seg_auc", "mean_seg_f1", "mean_pixel_ap", "mean_cls_auc", "mean_cls_f1"],
    )
    return summary_rows


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
