import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.utils import logging as transformers_logging

from map_refinement import refine_anomaly_map
from arc_refiner import load_mg_refiner, predict_patch_probs
from lec_calibrator import load_segad_calibrator, normalize_map, predict_segad_map


transformers_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME_OR_PATH = REPO_ROOT / "models" / "FGCLIP"
EVAL_SCRIPT_DIR = REPO_ROOT / "datasett" / "mvtec_ad_evaluation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "mvtec_fgclip2_benchmark"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasett"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESIZE_SHORT_EDGE = 1024
MAX_NUM_PATCHES = 4096
TOPK_RATIO = 0.002
GOOD_BANK_MAX_PATCHES = 50000
GOOD_BANK_PATCHES_PER_IMAGE = 256
GOOD_BANK_CHUNK_SIZE = 8192
DEFECT_TEXT_WEIGHT = 1.0
GOOD_TEXT_WEIGHT = 0.15
PROTO_DISTANCE_WEIGHT = 0.35
GOOD_BANK_DISTANCE_WEIGHT = 0.85
NORMAL_TEXT_WEIGHT = 0.25
NORMAL_PROTO_WEIGHT = 0.20
THRESHOLD_STD_MULT = 3.0
MIN_THRESHOLD_MARGIN = 0.015
PRO_INTEGRATION_LIMIT = 0.3
MG_MASK_RATIO = 0.10
MG_MASK_DILATE_RADIUS = 1
MG_FUSION_WEIGHT = 0.10
MG_FUSION_MODE = "positive"
MG_MASK_MODE = "ratio"
MG_ADAPTIVE_K = 1.0
MG_ADAPTIVE_QUANTILE = 0.90
MG_GATE_MIN_AREA = 0.001
MG_GATE_MAX_AREA = 0.3
MG_LOCAL_ONLY = True
MG_START_LAYER = 3
MG_MASK_THRESHOLD = 0.5
MG_NEG_BIAS = -1e4
MG_REFINER_THRESHOLD = 0.5
MAP_REFINE_MODE = "none"
MAP_REFINE_ALPHA = 0.0
MAP_REFINE_BG_SIGMA = 7.0
MAP_REFINE_TOPK_RATIO = 0.03
MAP_REFINE_CLAMP_QUANTILE = 1.0
PROGRESS_STYLE = "stage"


def set_progress_style(style: str):
    global PROGRESS_STYLE
    PROGRESS_STYLE = style


def iter_progress(iterable, desc: str, total: int | None = None):
    if PROGRESS_STYLE == "live":
        return tqdm(iterable, desc=desc, total=total, ascii=True, dynamic_ncols=True)

    count = total
    if count is None and hasattr(iterable, "__len__"):
        count = len(iterable)
    if count is None:
        print(f"[STAGE] {desc}")
    else:
        print(f"[STAGE] {desc} ({count} items)")

    def generator():
        for item in iterable:
            yield item
        print(f"[DONE] {desc}")

    return generator()

GENERIC_GOOD_PROMPTS = [
    "a defect-free industrial product",
    "an intact industrial object with clean surface",
    "a normal object with locally consistent appearance",
    "a uniform surface without local damage",
]

GENERIC_DEFECT_PROMPTS = [
    "a small localized defect on the surface",
    "a subtle local anomaly region",
    "a fine-grained damaged area",
    "a local structural defect",
    "a local appearance inconsistency",
]

OBJECT_PROMPTS = {
    "bottle": {
        "good": ["a normal bottle", "an intact bottle", "a clean bottle without defects"],
        "defect": [
            "a bottle with large breakage",
            "a bottle with small breakage",
            "a bottle with contamination",
            "a bottle with cracks",
        ],
    },
    "cable": {
        "good": [
            "a normal cable with continuous insulation",
            "an intact cable with aligned wires",
            "a cable without cut, break, or missing wire",
        ],
        "defect": [
            "a cable with a bent wire in a local region",
            "a cable with a missing wire strand",
            "a cable with a local cut on the insulation",
            "a cable with a small poke hole",
            "a cable with exposed inner wire",
        ],
    },
    "capsule": {
        "good": ["a normal capsule", "an intact capsule", "a clean capsule without defects"],
        "defect": [
            "a capsule with a crack",
            "a capsule with faulty imprint",
            "a capsule with a poke defect",
            "a capsule with a scratch",
            "a squeezed capsule with compression",
        ],
    },
    "carpet": {
        "good": ["a normal carpet", "a clean carpet", "a carpet without defects"],
        "defect": [
            "a carpet with a hole",
            "a carpet with color stain",
            "a carpet with metal contamination",
            "a carpet with thread residue",
            "a carpet with loose thread",
            "a carpet with a cut",
        ],
    },
    "grid": {
        "good": ["a normal grid", "an intact grid", "a regular metal grid without defects"],
        "defect": [
            "a grid with breakage",
            "a grid with thread residue",
            "a grid with loose thread",
            "a grid with metal contamination",
            "a grid with glue residue",
            "a grid with a bent shape",
        ],
    },
    "hazelnut": {
        "good": ["a normal hazelnut", "an intact hazelnut", "a healthy hazelnut surface"],
        "defect": [
            "a hazelnut with a crack",
            "a hazelnut with a cut",
            "a hazelnut with a hole",
            "a hazelnut with abnormal print",
        ],
    },
    "leather": {
        "good": ["normal leather", "clean leather", "leather without surface damage"],
        "defect": [
            "leather with color stain",
            "leather with a cut",
            "leather with a fold",
            "leather with glue residue",
            "leather with a poke defect",
        ],
    },
    "metal_nut": {
        "good": [
            "a normal metal nut with regular hexagonal contour",
            "an intact metal nut with centered inner hole",
            "a clean metal nut with complete thread and no deformation",
        ],
        "defect": [
            "a metal nut with bent or deformed contour",
            "a metal nut with color stain on the metal surface",
            "a metal nut with flipped orientation in the local view",
            "a metal nut with a local scratch on the rim",
            "a metal nut with damaged inner thread",
        ],
    },
    "pill": {
        "good": [
            "a normal pill with smooth coating",
            "an intact pill with regular imprint",
            "a clean pill without crack, stain, or contamination",
        ],
        "defect": [
            "a pill with color stain on a local region",
            "a pill with contamination on the surface",
            "a pill with a local crack",
            "a pill with faulty or abnormal imprint",
            "a pill with a small surface scratch",
            "a pill with abnormal type appearance",
        ],
    },
    "screw": {
        "good": [
            "a normal screw with intact head and regular thread",
            "an intact screw with clean metallic surface",
            "a screw without scratch, dent, or damaged front",
        ],
        "defect": [
            "a screw with damaged or manipulated front region",
            "a screw with a scratch on the neck",
            "a screw with a scratch on the head",
            "a screw with a local dent or scratch defect",
        ],
    },
    "tile": {
        "good": ["a normal tile", "a clean tile", "a tile without defects"],
        "defect": [
            "a tile with a crack",
            "a tile with glue strip",
            "a tile with gray stroke",
            "a tile with oil stain",
            "a tile with rough surface",
        ],
    },
    "toothbrush": {
        "good": [
            "a normal toothbrush with aligned bristles",
            "an intact toothbrush with regular bristle pattern",
            "a clean toothbrush without local damage",
        ],
        "defect": [
            "a toothbrush with abnormal bristle pattern",
            "a toothbrush with damaged or missing bristles",
            "a toothbrush with a local defect on the head",
        ],
    },
    "transistor": {
        "good": [
            "a normal transistor with straight aligned leads",
            "an intact transistor with complete metal pins",
            "a transistor without bent, cut, or misplaced lead",
        ],
        "defect": [
            "a transistor with a bent lead in a local region",
            "a transistor with a cut or missing lead",
            "a transistor with a misplaced or misaligned lead",
            "a transistor with local package surface damage",
            "a transistor with incomplete pin structure",
        ],
    },
    "wood": {
        "good": ["normal wood", "clean wood", "wood without defects"],
        "defect": [
            "wood with color stain",
            "wood with a hole",
            "wood with a scratch",
            "wood with liquid stain",
        ],
    },
    "zipper": {
        "good": [
            "a normal zipper with aligned teeth and intact fabric border",
            "an intact zipper with continuous tooth pattern",
            "a zipper without missing tooth, torn fabric, or split teeth",
        ],
        "defect": [
            "a zipper with broken or missing teeth",
            "a zipper with a torn fabric border defect",
            "a zipper with defective fabric near the edge",
            "a zipper with broken fabric in a local region",
            "a zipper with split teeth",
            "a zipper with squeezed or misaligned teeth",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FG-CLIP on the MVTec AD dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Path to the MVTec AD dataset root. "
            "It must contain folders like bottle/, cable/, etc. "
            f"Default: {DEFAULT_DATASET_ROOT}"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_NAME_OR_PATH,
        help="Local FG-CLIP model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write anomaly maps, metrics, and benchmark summary.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Subset of MVTec categories to evaluate. Default: all detected categories.",
    )
    parser.add_argument("--resize-short-edge", type=int, default=RESIZE_SHORT_EDGE)
    parser.add_argument("--max-num-patches", type=int, default=MAX_NUM_PATCHES)
    parser.add_argument("--topk-ratio", type=float, default=TOPK_RATIO)
    parser.add_argument("--good-bank-max-patches", type=int, default=GOOD_BANK_MAX_PATCHES)
    parser.add_argument("--good-bank-patches-per-image", type=int, default=GOOD_BANK_PATCHES_PER_IMAGE)
    parser.add_argument(
        "--good-bank-chunk-size",
        type=int,
        default=GOOD_BANK_CHUNK_SIZE,
        help="Chunk size for dense patch-to-good-bank similarity. Lower this if GPU memory is tight.",
    )
    parser.add_argument("--defect-text-weight", type=float, default=DEFECT_TEXT_WEIGHT)
    parser.add_argument("--good-text-weight", type=float, default=GOOD_TEXT_WEIGHT)
    parser.add_argument("--proto-distance-weight", type=float, default=PROTO_DISTANCE_WEIGHT)
    parser.add_argument("--good-bank-distance-weight", type=float, default=GOOD_BANK_DISTANCE_WEIGHT)
    parser.add_argument("--threshold-std-mult", type=float, default=THRESHOLD_STD_MULT)
    parser.add_argument("--min-threshold-margin", type=float, default=MIN_THRESHOLD_MARGIN)
    parser.add_argument("--pro-integration-limit", type=float, default=PRO_INTEGRATION_LIMIT)
    parser.add_argument(
        "--use-mg",
        action="store_true",
        help="Enable two-stage MG patch-mask refinement for dense image features.",
    )
    parser.add_argument(
        "--mg-mask-ratio",
        type=float,
        default=MG_MASK_RATIO,
        help="Ratio of highest-scoring first-stage patches kept by the MG mask.",
    )
    parser.add_argument(
        "--mg-mask-dilate-radius",
        type=int,
        default=MG_MASK_DILATE_RADIUS,
        help="Patch-neighborhood dilation radius for the MG mask. 0 disables dilation.",
    )
    parser.add_argument(
        "--mg-fusion-weight",
        type=float,
        default=MG_FUSION_WEIGHT,
        help="Fusion weight lambda for the MG refinement branch.",
    )
    parser.add_argument(
        "--mg-fusion-mode",
        choices=["none", "linear", "residual", "positive"],
        default=MG_FUSION_MODE,
        help=(
            "How to combine baseline and MG patch scores. none disables MG; "
            "linear/residual use ordinary interpolation; positive only preserves "
            "positive anomaly enhancement from MG branch."
        ),
    )
    parser.add_argument(
        "--mg-mask-mode",
        choices=["ratio", "adaptive"],
        default=MG_MASK_MODE,
        help="MG mask generation mode: top-k ratio or adaptive score threshold.",
    )
    parser.add_argument("--mg-adaptive-k", type=float, default=MG_ADAPTIVE_K)
    parser.add_argument("--mg-adaptive-quantile", type=float, default=MG_ADAPTIVE_QUANTILE)
    parser.add_argument(
        "--mg-use-gate",
        action="store_true",
        help="Enable reliability gate for positive residual MG fusion.",
    )
    parser.add_argument("--mg-gate-min-area", type=float, default=MG_GATE_MIN_AREA)
    parser.add_argument("--mg-gate-max-area", type=float, default=MG_GATE_MAX_AREA)
    parser.add_argument(
        "--mg-local-only",
        action=argparse.BooleanOptionalAction,
        default=MG_LOCAL_ONLY,
        help="Apply MG residual only inside the selected MG mask region.",
    )
    parser.add_argument("--mg-start-layer", type=int, default=MG_START_LAYER)
    parser.add_argument("--mg-mask-threshold", type=float, default=MG_MASK_THRESHOLD)
    parser.add_argument("--mg-neg-bias", type=float, default=MG_NEG_BIAS)
    parser.add_argument(
        "--mg-refiner-checkpoint",
        type=Path,
        default=None,
        help="Optional trained MG patch refiner checkpoint. If unavailable, falls back to score-based masks.",
    )
    parser.add_argument(
        "--mg-refiner-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-category MG refiner checkpoints named <object>.pt.",
    )
    parser.add_argument(
        "--mg-refiner-threshold",
        type=float,
        default=MG_REFINER_THRESHOLD,
        help="Patch probability threshold used by the trained MG refiner.",
    )
    parser.add_argument(
        "--mg-refiner-mask-mode",
        choices=["threshold", "topk", "score_intersect"],
        default="score_intersect",
        help=(
            "How the trained MG refiner selects patches. score_intersect is conservative: "
            "a patch must be selected by both refiner probability and baseline anomaly score."
        ),
    )
    parser.add_argument(
        "--mg-refiner-topk-ratio",
        type=float,
        default=0.05,
        help="Top-k ratio for refiner probability mask when using topk or score_intersect.",
    )
    parser.add_argument(
        "--mg-refiner-score-ratio",
        type=float,
        default=0.05,
        help="Top-k ratio for baseline-score prior when using score_intersect.",
    )
    parser.add_argument(
        "--mg-refiner-soft-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use refiner probabilities as a soft local gate for the positive residual.",
    )
    parser.add_argument(
        "--mg-refiner-score-boost",
        type=float,
        default=0.0,
        help=(
            "Localization-only boost from trained refiner probabilities. "
            "0 keeps the original positive residual behavior."
        ),
    )
    parser.add_argument(
        "--mg-refiner-score-power",
        type=float,
        default=1.0,
        help="Power applied to refiner probability confidence before score boosting.",
    )
    parser.add_argument(
        "--mg-use-refiner-weight",
        action="store_true",
        help="Use the learned scalar fusion weight stored in the MG refiner checkpoint.",
    )
    parser.add_argument("--segad-calibrator", type=Path, default=None)
    parser.add_argument(
        "--segad-calibrator-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-category SegAD calibrators named <object>.pkl.",
    )
    parser.add_argument(
        "--segad-blend-weight",
        type=float,
        default=0.0,
        help="Blend weight for SegAD-style calibrated probability map. 0 disables it.",
    )
    parser.add_argument(
        "--segad-fusion-mode",
        choices=["linear", "positive"],
        default="positive",
        help="positive only adds confident local calibration residuals; linear keeps the older map blending.",
    )
    parser.add_argument("--segad-power", type=float, default=1.0)
    parser.add_argument(
        "--segad-min-confidence",
        type=float,
        default=0.55,
        help="Minimum SegAD probability required before positive calibration can boost a pixel.",
    )
    parser.add_argument(
        "--image-score-source",
        choices=["final", "pre_segad"],
        default="final",
        help="Use final map or the pre-SegAD map to compute the per-image score recorded in summaries.",
    )
    parser.add_argument(
        "--classification-score-source",
        choices=["map", "image_score"],
        default="map",
        help="Use saved anomaly maps or recorded image scores for image-level AUROC/pAUROC.",
    )
    parser.add_argument(
        "--map-refine-mode",
        choices=["none", "local_contrast", "topk_contrast"],
        default=MAP_REFINE_MODE,
        help="Optional positive local-contrast enhancement for anomaly maps.",
    )
    parser.add_argument("--map-refine-alpha", type=float, default=MAP_REFINE_ALPHA)
    parser.add_argument("--map-refine-bg-sigma", type=float, default=MAP_REFINE_BG_SIGMA)
    parser.add_argument("--map-refine-topk-ratio", type=float, default=MAP_REFINE_TOPK_RATIO)
    parser.add_argument("--map-refine-clamp-quantile", type=float, default=MAP_REFINE_CLAMP_QUANTILE)
    parser.add_argument("--progress_style", choices=["stage", "live"], default="stage")
    return parser.parse_args()


def ensure_eval_importable():
    eval_dir = str(EVAL_SCRIPT_DIR)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    try:
        import generic_util as eval_util  # noqa: F401
        from evaluate_experiment import calculate_au_pro_au_roc, parse_dataset_files  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to import MVTec evaluation scripts from {EVAL_SCRIPT_DIR}. "
            "Please keep datasett/mvtec_ad_evaluation intact."
        ) from exc

    def read_tiff_windows_safe(file_path_no_ext, exts=(".tif", ".tiff", ".TIF", ".TIFF")):
        """Windows-safe variant of the official helper.

        On Windows, `os.path.exists("foo.tiff")` and `os.path.exists("foo.TIFF")`
        both return True for the same file. The original helper counts those as
        two different matches and raises a false duplicate-file error.
        """
        seen = {}
        for ext in exts:
            candidate = file_path_no_ext + ext
            if Path(candidate).exists():
                seen[str(Path(candidate)).lower()] = candidate

        file_paths = list(seen.values())
        if len(file_paths) == 0:
            raise FileNotFoundError(
                "Could not find a file with a TIFF extension"
                f" at {file_path_no_ext}"
            )
        if len(file_paths) > 1:
            raise IOError(
                "Found multiple files with a TIFF extension at"
                f" {file_path_no_ext}\n"
                "Please keep only one of .tif/.tiff for each anomaly map."
            )
        return tifffile.imread(file_paths[0])

    eval_util.read_tiff = read_tiff_windows_safe
    return eval_util, parse_dataset_files, calculate_au_pro_au_roc


def l2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def resize_short_edge(image: Image.Image, target_size: int):
    width, height = image.size
    short_edge = min(width, height)
    if short_edge >= target_size:
        return image
    scale = target_size / short_edge
    return image.resize((int(width * scale), int(height * scale)), Image.BICUBIC)


def gaussian_blur_map(array: np.ndarray, sigma: float = 1.2):
    if sigma <= 0:
        return array.astype(np.float32, copy=False)

    radius = max(1, int(round(3 * sigma)))
    kernel_size = 2 * radius + 1
    coords = torch.arange(kernel_size, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    tensor = torch.from_numpy(array.astype(np.float32, copy=False)).view(1, 1, array.shape[0], array.shape[1])
    blurred = F.conv2d(tensor, kernel_2d, padding=radius)
    return blurred[0, 0].cpu().numpy()


def standardize_map(raw_map: np.ndarray, mean_value: float, std_value: float, eps: float = 1e-6):
    scale = std_value if std_value > eps else eps
    return ((raw_map - mean_value) / scale).astype(np.float32, copy=False)


def sample_good_memory_tokens(dense_feat: torch.Tensor, per_image_limit: int):
    if per_image_limit <= 0 or dense_feat.shape[0] <= per_image_limit:
        return dense_feat
    indices = torch.linspace(0, dense_feat.shape[0] - 1, steps=per_image_limit, device=dense_feat.device)
    indices = indices.round().long().unique(sorted=True)
    return dense_feat.index_select(0, indices)


def finalize_good_memory_bank(memory_tokens, max_patches: int):
    if not memory_tokens:
        raise RuntimeError("Failed to build good memory bank: no sampled patches found.")

    memory_bank = torch.cat(memory_tokens, dim=0)
    if max_patches > 0 and memory_bank.shape[0] > max_patches:
        indices = torch.linspace(0, memory_bank.shape[0] - 1, steps=max_patches, device=memory_bank.device)
        indices = indices.round().long().unique(sorted=True)
        memory_bank = memory_bank.index_select(0, indices)
    return l2norm(memory_bank, dim=-1)


def compute_nearest_good_similarity(dense_feat: torch.Tensor, good_memory_bank: torch.Tensor, chunk_size: int):
    if good_memory_bank is None or good_memory_bank.shape[0] == 0:
        return None

    if chunk_size <= 0 or good_memory_bank.shape[0] <= chunk_size:
        return (dense_feat @ good_memory_bank.T).max(dim=1).values

    best_similarity = None
    for start in range(0, good_memory_bank.shape[0], chunk_size):
        bank_chunk = good_memory_bank[start : start + chunk_size]
        chunk_best = (dense_feat @ bank_chunk.T).max(dim=1).values
        best_similarity = chunk_best if best_similarity is None else torch.maximum(best_similarity, chunk_best)
    return best_similarity


def compute_binary_f1_max(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    unique_thresholds = np.unique(scores)

    best_f1 = 0.0
    best_threshold = None
    best_precision = 0.0
    best_recall = 0.0

    for threshold in unique_thresholds:
        preds = scores >= threshold
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)

    return {
        "f1_max": float(best_f1),
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
    }


def flatten_eval_arrays(ground_truth, predictions):
    labels = np.concatenate([gt.reshape(-1) for gt in ground_truth]).astype(np.uint8, copy=False)
    scores = np.concatenate([pred.reshape(-1) for pred in predictions]).astype(np.float32, copy=False)
    return scores, labels


def compute_binary_auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
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


def compute_binary_f1_max_fast(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    num_pos = int(np.sum(labels == 1))
    if num_pos == 0:
        return {
            "f1_max": 0.0,
            "best_threshold": None,
            "best_precision": 0.0,
            "best_recall": 0.0,
        }

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    distinct_mask = np.r_[np.diff(sorted_scores) != 0, True]
    tp = np.cumsum(sorted_labels == 1)[distinct_mask].astype(np.float64)
    fp = np.cumsum(sorted_labels == 0)[distinct_mask].astype(np.float64)
    thresholds = sorted_scores[distinct_mask]

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / num_pos
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    best_idx = int(np.argmax(f1))
    return {
        "f1_max": float(f1[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
    }


def compute_classification_roc_from_scores(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    assert scores.shape[0] == labels.shape[0]

    num_pos = int(np.sum(labels != 0))
    num_neg = int(np.sum(labels == 0))
    if num_pos == 0 or num_neg == 0:
        return [0.0, 1.0], [0.0, 1.0]

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    fprs = [1.0]
    tprs = [1.0]
    num_fp = num_neg
    num_tp = num_pos

    for idx, (score, label_value) in enumerate(zip(sorted_scores, sorted_labels)):
        if label_value == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        next_score = sorted_scores[idx + 1] if idx < len(sorted_scores) - 1 else None
        if next_score is None or next_score != score:
            fprs.append(num_fp / num_neg)
            tprs.append(num_tp / num_pos)

    return fprs[::-1], tprs[::-1]


def load_eval_arrays(eval_util, gt_filenames, prediction_filenames):
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    for gt_name, pred_name in iter_progress(zip(gt_filenames, prediction_filenames), desc="load eval arrays", total=len(gt_filenames)):
        prediction = eval_util.read_tiff(pred_name).astype(np.float32, copy=False)
        predictions.append(prediction)

        if gt_name is not None:
            gt = (np.asarray(Image.open(gt_name)) > 0).astype(np.uint8)
        else:
            gt = np.zeros(prediction.shape, dtype=np.uint8)
        ground_truth.append(gt)

    return ground_truth, predictions


def load_model(model_path: Path):
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    return model, tokenizer, image_processor


def attach_mg_refiner(args):
    args._mg_refiner = None
    args._mg_refiner_metadata = {}
    args._mg_refiner_fusion_weight = None
    args._mg_refiner_cache = {}
    args._mg_refiner_dir_warned = set()
    checkpoint = getattr(args, "mg_refiner_checkpoint", None)
    if checkpoint is None:
        return

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        print(f"[WARN] MG refiner checkpoint not found, using score-based MG mask: {checkpoint}")
        return

    try:
        refiner, metadata = load_mg_refiner(checkpoint, device=DEVICE)
    except Exception as exc:  # pragma: no cover - runtime safety for optional branch
        print(f"[WARN] Failed to load MG refiner, using score-based MG mask. Reason: {exc}")
        return

    args._mg_refiner = refiner
    args._mg_refiner_metadata = metadata
    args._mg_refiner_fusion_weight = float(refiner.fusion_weight().detach().cpu().item())
    print(f"[INFO] mg_refiner_checkpoint = {checkpoint}")
    print(f"[INFO] mg_refiner_weight     = {args._mg_refiner_fusion_weight:.4f}")


def attach_segad_calibrator(args):
    args._segad_calibrator = None
    args._segad_calibrator_metadata = {}
    args._segad_calibrator_cache = {}
    args._segad_calibrator_dir_warned = set()
    calibrator_path = getattr(args, "segad_calibrator", None)
    if calibrator_path is None:
        return

    calibrator_path = Path(calibrator_path).resolve()
    if not calibrator_path.exists():
        print(f"[WARN] SegAD calibrator not found: {calibrator_path}")
        return
    calibrator, metadata = load_segad_calibrator(calibrator_path)
    args._segad_calibrator = calibrator
    args._segad_calibrator_metadata = metadata
    print(f"[INFO] segad_calibrator = {calibrator_path}")


def activate_object_mg_refiner(args, object_name: str):
    refiner_dir = getattr(args, "mg_refiner_dir", None)
    if refiner_dir is None:
        return

    checkpoint = Path(refiner_dir).resolve() / f"{object_name}.pt"
    if not checkpoint.exists():
        warned = getattr(args, "_mg_refiner_dir_warned", set())
        if object_name not in warned:
            print(f"[WARN] Per-object MG refiner not found for {object_name}: {checkpoint}")
            warned.add(object_name)
            args._mg_refiner_dir_warned = warned
        return

    cache = getattr(args, "_mg_refiner_cache", {})
    if object_name not in cache:
        refiner, metadata = load_mg_refiner(checkpoint, device=DEVICE)
        cache[object_name] = (
            refiner,
            metadata,
            float(refiner.fusion_weight().detach().cpu().item()),
            checkpoint,
        )
        args._mg_refiner_cache = cache

    refiner, metadata, fusion_weight, checkpoint = cache[object_name]
    args._mg_refiner = refiner
    args._mg_refiner_metadata = metadata
    args._mg_refiner_fusion_weight = fusion_weight
    print(f"[INFO] mg_refiner[{object_name}] = {checkpoint}")


def activate_object_segad_calibrator(args, object_name: str):
    calibrator_dir = getattr(args, "segad_calibrator_dir", None)
    if calibrator_dir is None:
        return

    calibrator_path = Path(calibrator_dir).resolve() / f"{object_name}.pkl"
    if not calibrator_path.exists():
        warned = getattr(args, "_segad_calibrator_dir_warned", set())
        if object_name not in warned:
            print(f"[WARN] SegAD calibrator not found for {object_name}: {calibrator_path}")
            warned.add(object_name)
            args._segad_calibrator_dir_warned = warned
        return

    cache = getattr(args, "_segad_calibrator_cache", {})
    if object_name not in cache:
        cache[object_name] = (*load_segad_calibrator(calibrator_path), calibrator_path)
        args._segad_calibrator_cache = cache

    calibrator, metadata, _ = cache[object_name]
    args._segad_calibrator = calibrator
    args._segad_calibrator_metadata = metadata
    print(f"[INFO] segad_calibrator[{object_name}] = {calibrator_path}")


def describe_device():
    if DEVICE != "cuda":
        return DEVICE
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024**3)
    return f"cuda:0 ({props.name}, {total_gib:.1f} GiB)"


@torch.inference_mode()
def encode_text(prompts, tokenizer, model):
    max_length = int(model.config.text_config.max_position_embeddings)
    text_inputs = tokenizer(
        [prompt.lower() for prompt in prompts],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    position_ids = torch.arange(max_length, device=DEVICE).expand(text_inputs["input_ids"].shape[0], -1)
    text_feat = model.get_text_features(**text_inputs, position_ids=position_ids, walk_type="box")
    return l2norm(text_feat)


def build_mg_patch_mask(
    patch_scores: torch.Tensor,
    real_h: int,
    real_w: int,
    max_tokens: int,
    ratio: float,
    dilate_radius: int,
    mask_mode: str = MG_MASK_MODE,
    adaptive_k: float = MG_ADAPTIVE_K,
    adaptive_quantile: float = MG_ADAPTIVE_QUANTILE,
):
    real_tokens = real_h * real_w
    patch_scores = patch_scores.detach().float().reshape(-1)
    if patch_scores.numel() != real_tokens:
        raise RuntimeError(
            f"MG patch score length mismatch: got {patch_scores.numel()}, expected {real_tokens}."
        )

    real_mask = torch.zeros(real_tokens, device=patch_scores.device, dtype=torch.float32)
    if mask_mode == "adaptive":
        quantile = min(max(float(adaptive_quantile), 0.0), 1.0)
        mean_threshold = patch_scores.mean() + float(adaptive_k) * patch_scores.std(unbiased=False)
        quantile_threshold = torch.quantile(patch_scores, quantile)
        threshold = torch.maximum(mean_threshold, quantile_threshold)
        real_mask = (patch_scores >= threshold).float()
        if real_mask.sum() < 1:
            real_mask[torch.argmax(patch_scores)] = 1.0
    else:
        ratio = min(max(float(ratio), 0.0), 1.0)
        keep = max(1, int(round(real_tokens * ratio)))
        keep = min(keep, real_tokens)
        top_indices = torch.topk(patch_scores, k=keep, largest=True).indices
        real_mask[top_indices] = 1.0

    return build_mg_patch_mask_from_real_mask(real_mask, real_h, real_w, max_tokens, dilate_radius)


def build_mg_patch_mask_from_real_mask(
    real_mask: torch.Tensor,
    real_h: int,
    real_w: int,
    max_tokens: int,
    dilate_radius: int,
):
    real_tokens = real_h * real_w
    real_mask = real_mask.detach().float().reshape(-1)
    if real_mask.numel() != real_tokens:
        raise RuntimeError(
            f"MG mask length mismatch: got {real_mask.numel()}, expected {real_tokens}."
        )

    if dilate_radius > 0:
        mask_2d = real_mask.view(1, 1, real_h, real_w)
        kernel_size = 2 * int(dilate_radius) + 1
        mask_2d = F.max_pool2d(mask_2d, kernel_size=kernel_size, stride=1, padding=int(dilate_radius))
        real_mask = mask_2d.view(-1)

    area_ratio = float(real_mask.mean().detach().cpu().item())

    full_mask = torch.zeros(max_tokens, device=real_mask.device, dtype=torch.float32)
    full_mask[:real_tokens] = real_mask
    return full_mask.unsqueeze(0), area_ratio


def select_topk_mask(values: torch.Tensor, ratio: float):
    values = values.detach().float().reshape(-1)
    ratio = min(max(float(ratio), 0.0), 1.0)
    keep = max(1, int(round(values.numel() * ratio)))
    keep = min(keep, values.numel())
    mask = torch.zeros_like(values)
    mask[torch.topk(values, k=keep, largest=True).indices] = 1.0
    return mask


def build_refiner_mg_patch_mask(
    dense_feat: torch.Tensor,
    patch_scores: torch.Tensor,
    real_h: int,
    real_w: int,
    max_tokens: int,
    args,
):
    refiner = getattr(args, "_mg_refiner", None)
    if refiner is None:
        raise RuntimeError("MG refiner is not loaded.")

    patch_probs = predict_patch_probs(refiner, dense_feat, patch_scores).to(device=patch_scores.device)
    mask_mode = getattr(args, "mg_refiner_mask_mode", "score_intersect")

    if mask_mode == "threshold":
        threshold = min(max(float(args.mg_refiner_threshold), 0.0), 1.0)
        real_mask = (patch_probs >= threshold).float()
    elif mask_mode == "topk":
        real_mask = select_topk_mask(patch_probs, args.mg_refiner_topk_ratio)
    else:
        prob_mask = select_topk_mask(patch_probs, args.mg_refiner_topk_ratio)
        score_mask = select_topk_mask(patch_scores, args.mg_refiner_score_ratio)
        real_mask = prob_mask * score_mask
        if real_mask.sum() < 1:
            score_std = patch_scores.detach().float().std(unbiased=False).clamp_min(1e-6)
            score_prior = torch.sigmoid((patch_scores.detach().float() - patch_scores.detach().float().mean()) / score_std)
            real_mask = select_topk_mask(patch_probs * score_prior, args.mg_refiner_topk_ratio)

    if real_mask.sum() < 1:
        real_mask[torch.argmax(patch_probs)] = 1.0

    mg_patch_mask, area_ratio = build_mg_patch_mask_from_real_mask(
        real_mask=real_mask,
        real_h=real_h,
        real_w=real_w,
        max_tokens=max_tokens,
        dilate_radius=args.mg_mask_dilate_radius,
    )
    fusion_real_mask = real_mask
    if getattr(args, "mg_refiner_soft_fusion", True):
        # Soft local gate keeps the positive residual strongest where the learned refiner is confident.
        fusion_real_mask = real_mask * patch_probs.detach().float().clamp(0.0, 1.0)
    mg_fusion_mask, _ = build_mg_patch_mask_from_real_mask(
        real_mask=fusion_real_mask,
        real_h=real_h,
        real_w=real_w,
        max_tokens=max_tokens,
        dilate_radius=args.mg_mask_dilate_radius,
    )
    return mg_patch_mask, area_ratio, mg_fusion_mask, patch_probs.detach().float()


def compute_mg_gate(area_ratio: float, min_area: float, max_area: float):
    if area_ratio < min_area:
        return 0.3
    if area_ratio > max_area:
        return 0.5
    return 1.0


def fuse_patch_scores(
    base_score: torch.Tensor,
    mg_score: torch.Tensor,
    mode: str,
    weight: float,
    gate: float = 1.0,
    fusion_mask: torch.Tensor = None,
):
    if base_score.shape != mg_score.shape:
        raise RuntimeError(f"MG fusion shape mismatch: base={tuple(base_score.shape)}, mg={tuple(mg_score.shape)}")

    base_score = base_score.float()
    mg_score = mg_score.to(device=base_score.device, dtype=base_score.dtype)
    weight = min(max(float(weight), 0.0), 1.0)
    gate = min(max(float(gate), 0.0), 1.0)
    if fusion_mask is None:
        fusion_mask = 1.0
    else:
        if fusion_mask.shape != base_score.shape:
            raise RuntimeError(
                f"MG fusion mask shape mismatch: mask={tuple(fusion_mask.shape)}, score={tuple(base_score.shape)}"
            )
        fusion_mask = fusion_mask.to(device=base_score.device, dtype=base_score.dtype).clamp(0.0, 1.0)

    if mode == "none":
        fused = base_score
    elif mode == "linear" or mode == "residual":
        fused = base_score + fusion_mask * weight * gate * (mg_score - base_score)
    elif mode == "positive":
        # Only preserves positive anomaly enhancement from MG branch.
        fused = base_score + fusion_mask * weight * gate * torch.relu(mg_score - base_score)
    else:
        raise ValueError(f"Unsupported MG fusion mode: {mode}")

    return torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6).clamp(min=-1e6, max=1e6)


def apply_refiner_score_boost(
    base_score: torch.Tensor,
    mg_score: torch.Tensor,
    refiner_probs: torch.Tensor,
    threshold: float,
    boost: float,
    power: float,
):
    boost = max(float(boost), 0.0)
    if boost <= 0:
        return mg_score

    base_score = base_score.float()
    mg_score = mg_score.to(device=base_score.device, dtype=base_score.dtype)
    refiner_probs = refiner_probs.to(device=base_score.device, dtype=base_score.dtype).reshape(-1)
    if refiner_probs.shape != base_score.shape:
        raise RuntimeError(
            f"MG refiner boost shape mismatch: probs={tuple(refiner_probs.shape)}, score={tuple(base_score.shape)}"
        )

    threshold = min(max(float(threshold), 0.0), 0.99)
    denom = max(1.0 - threshold, 1e-6)
    confidence = ((refiner_probs - threshold) / denom).clamp(0.0, 1.0)
    confidence = confidence.pow(max(float(power), 0.1))
    local_scale = base_score.std(unbiased=False).clamp_min(1e-6)
    boosted_score = base_score + boost * local_scale * confidence
    return torch.maximum(mg_score, boosted_score)


def apply_segad_calibration(raw_map: np.ndarray, args):
    calibrator = getattr(args, "_segad_calibrator", None)
    blend_weight = min(max(float(getattr(args, "segad_blend_weight", 0.0)), 0.0), 1.0)
    if calibrator is None or blend_weight <= 0:
        return raw_map

    prob_map = predict_segad_map(raw_map, calibrator)
    power = max(float(getattr(args, "segad_power", 1.0)), 0.1)
    prob_map = np.power(np.clip(prob_map, 0.0, 1.0), power).astype(np.float32)
    base_map = normalize_map(raw_map)
    if getattr(args, "segad_fusion_mode", "positive") == "linear":
        return ((1.0 - blend_weight) * base_map + blend_weight * prob_map).astype(np.float32)

    min_confidence = min(max(float(getattr(args, "segad_min_confidence", 0.55)), 0.0), 0.99)
    residual = np.maximum(prob_map - base_map, 0.0)
    residual = np.where(prob_map >= min_confidence, residual, 0.0).astype(np.float32)
    # SegAD positive fusion only preserves confident local anomaly enhancement
    # from the supervised calibration branch, instead of replacing the baseline map.
    scale = float(np.std(raw_map)) + 1e-6
    return (raw_map.astype(np.float32, copy=False) + blend_weight * scale * residual).astype(np.float32)


@torch.inference_mode()
def encode_dense_image(
    image: Image.Image,
    image_processor,
    model,
    resize_target: int,
    max_num_patches: int,
    mg_patch_mask: torch.Tensor = None,
    mg_start_layer: int = MG_START_LAYER,
    mg_end_layer: int | None = None,
    mg_mask_threshold: float = MG_MASK_THRESHOLD,
    mg_neg_bias: float = MG_NEG_BIAS,
    attention_bias_eta: float = 1.0,
    feature_layer: int = 0,
):
    image_for_model = resize_short_edge(image.convert("RGB"), resize_target)
    image_inputs = image_processor(
        images=image_for_model,
        max_num_patches=max_num_patches,
        return_tensors="pt",
    ).to(DEVICE)

    feature_layer = int(feature_layer or 0)

    if feature_layer > 0 and mg_patch_mask is not None:
        raise RuntimeError("Intermediate feature extraction does not support MG patch masking.")

    if feature_layer > 0:
        if not hasattr(model, "vision_model"):
            raise RuntimeError("Current FGCLIP model does not expose vision_model for intermediate extraction.")
        vision_model = model.vision_model
        layers = getattr(vision_model.encoder, "layers", None)
        if layers is None:
            raise RuntimeError("Current FGCLIP vision encoder does not expose encoder layers.")
        if feature_layer < 1 or feature_layer > len(layers):
            raise ValueError(f"feature_layer must be in [1, {len(layers)}], got {feature_layer}")

        pixel_values = image_inputs["pixel_values"]
        spatial_shapes_inputs = image_inputs["spatial_shapes"]
        pixel_attention_mask = image_inputs.get("pixel_attention_mask", None)
        hidden_states = vision_model.embeddings(pixel_values, spatial_shapes_inputs)
        if pixel_attention_mask is not None and vision_model.config._attn_implementation != "flash_attention_2":
            attention_mask = _prepare_4d_attention_mask(pixel_attention_mask, hidden_states.dtype)
        else:
            attention_mask = pixel_attention_mask

        for index, encoder_layer in enumerate(layers, start=1):
            hidden_states = encoder_layer(hidden_states, attention_mask)
            if index == feature_layer:
                break
        dense_feat = vision_model.post_layernorm(hidden_states)
    elif mg_patch_mask is None:
        dense_feat = model.get_image_dense_feature(**image_inputs)
    else:
        dense_feat = model.get_image_dense_feature(
            **image_inputs,
            mg_patch_mask=mg_patch_mask.to(DEVICE),
            mg_start_layer=mg_start_layer,
            mg_end_layer=mg_end_layer,
            mg_mask_threshold=mg_mask_threshold,
            mg_neg_bias=mg_neg_bias,
            attention_bias_eta=attention_bias_eta,
        )
    spatial_shapes = image_inputs["spatial_shapes"][0]
    real_h = int(spatial_shapes[0].item())
    real_w = int(spatial_shapes[1].item())
    real_tokens = real_h * real_w

    dense_feat = dense_feat[0][:real_tokens]
    return l2norm(dense_feat), real_h, real_w, int(image_inputs["pixel_values"].shape[1])


def get_prompts(object_name: str):
    prompt_cfg = OBJECT_PROMPTS[object_name]
    good_prompts = prompt_cfg["good"] + GENERIC_GOOD_PROMPTS
    defect_prompts = prompt_cfg["defect"] + GENERIC_DEFECT_PROMPTS
    return good_prompts, defect_prompts


@torch.inference_mode()
def build_good_prototype(good_dir: Path, image_processor, model, args):
    good_paths = list_images(good_dir)
    if not good_paths:
        raise RuntimeError(f"No training good images found in {good_dir}")

    good_vecs = []
    memory_tokens = []
    for img_path in iter_progress(good_paths, desc=f"build prototype {good_dir.parent.name}"):
        image = Image.open(img_path).convert("RGB")
        dense_feat, _, _, _ = encode_dense_image(
            image=image,
            image_processor=image_processor,
            model=model,
            resize_target=args.resize_short_edge,
            max_num_patches=args.max_num_patches,
            feature_layer=getattr(args, "feature_layer", 0),
        )
        img_vec = l2norm(dense_feat.mean(dim=0), dim=-1)
        good_vecs.append(img_vec)
        memory_tokens.append(sample_good_memory_tokens(dense_feat, args.good_bank_patches_per_image))

    good_proto = l2norm(torch.stack(good_vecs, dim=0).mean(dim=0), dim=-1)
    good_memory_bank = finalize_good_memory_bank(memory_tokens, args.good_bank_max_patches)
    return good_proto, good_memory_bank, good_paths


@torch.inference_mode()
def compute_maps_and_score(
    image,
    image_processor,
    model,
    good_proto,
    good_memory_bank,
    good_text_bank,
    defect_text_bank,
    args,
):
    dense_feat, real_h, real_w, max_tokens = encode_dense_image(
        image=image,
        image_processor=image_processor,
        model=model,
        resize_target=args.resize_short_edge,
        max_num_patches=args.max_num_patches,
        feature_layer=getattr(args, "feature_layer", 0),
    )

    def score_dense_features(features: torch.Tensor):
        defect_score = (features @ defect_text_bank.T).max(dim=1).values
        good_text_score = (features @ good_text_bank.T).max(dim=1).values
        good_proto_score = features @ good_proto
        proto_distance = 1.0 - good_proto_score

        if good_memory_bank is not None and good_memory_bank.shape[0] > 0:
            nearest_good_similarity = compute_nearest_good_similarity(
                features,
                good_memory_bank,
                args.good_bank_chunk_size,
            )
            good_bank_distance = 1.0 - nearest_good_similarity
        else:
            good_bank_distance = torch.zeros_like(proto_distance)

        return (
            args.defect_text_weight * defect_score
            - args.good_text_weight * good_text_score
            + args.proto_distance_weight * proto_distance
            + args.good_bank_distance_weight * good_bank_distance
        )

    raw_patch_score = score_dense_features(dense_feat)

    if args.use_mg and args.mg_fusion_mode != "none":
        try:
            mg_refiner_probs = None
            if getattr(args, "_mg_refiner", None) is not None:
                mg_patch_mask, mg_area_ratio, mg_fusion_mask, mg_refiner_probs = build_refiner_mg_patch_mask(
                    dense_feat=dense_feat,
                    patch_scores=raw_patch_score,
                    real_h=real_h,
                    real_w=real_w,
                    max_tokens=max_tokens,
                    args=args,
                )
            else:
                mg_patch_mask, mg_area_ratio = build_mg_patch_mask(
                    raw_patch_score,
                    real_h=real_h,
                    real_w=real_w,
                    max_tokens=max_tokens,
                    ratio=args.mg_mask_ratio,
                    dilate_radius=args.mg_mask_dilate_radius,
                    mask_mode=args.mg_mask_mode,
                    adaptive_k=args.mg_adaptive_k,
                    adaptive_quantile=args.mg_adaptive_quantile,
                )
                mg_fusion_mask = mg_patch_mask

            mg_dense_feat, mg_real_h, mg_real_w, _ = encode_dense_image(
                image=image,
                image_processor=image_processor,
                model=model,
                resize_target=args.resize_short_edge,
                max_num_patches=args.max_num_patches,
                mg_patch_mask=mg_patch_mask,
                mg_start_layer=args.mg_start_layer,
                mg_end_layer=getattr(args, "mg_end_layer", None),
                mg_mask_threshold=args.mg_mask_threshold,
                mg_neg_bias=args.mg_neg_bias,
                attention_bias_eta=getattr(args, "attention_bias_eta", 1.0),
            )
            if (mg_real_h, mg_real_w) != (real_h, real_w):
                raise RuntimeError(
                    f"MG feature shape mismatch: base={(real_h, real_w)}, mg={(mg_real_h, mg_real_w)}"
                )
            mg_patch_score = score_dense_features(mg_dense_feat)
            if mg_refiner_probs is not None:
                mg_patch_score = apply_refiner_score_boost(
                    base_score=raw_patch_score,
                    mg_score=mg_patch_score,
                    refiner_probs=mg_refiner_probs,
                    threshold=args.mg_refiner_threshold,
                    boost=args.mg_refiner_score_boost,
                    power=args.mg_refiner_score_power,
                )
            gate = (
                compute_mg_gate(mg_area_ratio, args.mg_gate_min_area, args.mg_gate_max_area)
                if args.mg_use_gate
                else 1.0
            )
            fusion_weight = (
                getattr(args, "_mg_refiner_fusion_weight", args.mg_fusion_weight)
                if getattr(args, "mg_use_refiner_weight", False)
                else args.mg_fusion_weight
            )
            raw_patch_score = fuse_patch_scores(
                base_score=raw_patch_score,
                mg_score=mg_patch_score,
                mode=args.mg_fusion_mode,
                weight=fusion_weight,
                gate=gate,
                fusion_mask=mg_fusion_mask[0, : raw_patch_score.numel()] if args.mg_local_only else None,
            )
        except (TypeError, RuntimeError, AttributeError) as exc:
            if not getattr(args, "_mg_fallback_warned", False):
                print(f"[WARN] MG branch failed and will fall back to baseline. Reason: {exc}")
                args._mg_fallback_warned = True

    raw_map = raw_patch_score.reshape(real_h, real_w).detach().float().cpu().numpy()
    raw_map = gaussian_blur_map(raw_map, sigma=1.2)
    pre_segad_map = raw_map
    raw_map = apply_segad_calibration(raw_map, args)
    raw_map = refine_anomaly_map(
        raw_map,
        mode=args.map_refine_mode,
        alpha=args.map_refine_alpha,
        bg_sigma=args.map_refine_bg_sigma,
        topk_ratio=args.map_refine_topk_ratio,
        clamp_quantile=args.map_refine_clamp_quantile,
    )

    score_map = pre_segad_map if getattr(args, "image_score_source", "final") == "pre_segad" else raw_map
    flat = score_map.reshape(-1)
    k = max(1, int(len(flat) * args.topk_ratio))
    image_score = float(np.mean(np.sort(flat)[-k:]))

    mn, mx = raw_map.min(), raw_map.max()
    norm_map = (raw_map - mn) / (mx - mn + 1e-8)
    return raw_map, norm_map, image_score


@torch.inference_mode()
def calibrate_threshold(
    good_paths,
    image_processor,
    model,
    good_proto,
    good_memory_bank,
    good_text_bank,
    defect_text_bank,
    args,
):
    scores = []
    raw_sum = 0.0
    raw_sq_sum = 0.0
    raw_count = 0

    for img_path in iter_progress(good_paths, desc="calibrate threshold"):
        image = Image.open(img_path).convert("RGB")
        raw_map, _, score = compute_maps_and_score(
            image=image,
            image_processor=image_processor,
            model=model,
            good_proto=good_proto,
            good_memory_bank=good_memory_bank,
            good_text_bank=good_text_bank,
            defect_text_bank=defect_text_bank,
            args=args,
        )
        scores.append(score)

        flat = raw_map.reshape(-1).astype(np.float64, copy=False)
        raw_sum += float(flat.sum())
        raw_sq_sum += float(np.square(flat).sum())
        raw_count += int(flat.size)

    scores = np.asarray(scores, dtype=np.float32)
    mean_score = float(scores.mean())
    std_score = float(scores.std())
    threshold = mean_score + max(args.threshold_std_mult * std_score, args.min_threshold_margin)

    raw_mean = raw_sum / max(raw_count, 1)
    raw_var = max(raw_sq_sum / max(raw_count, 1) - raw_mean * raw_mean, 0.0)
    raw_std = raw_var ** 0.5

    return threshold, mean_score, std_score, scores.tolist(), raw_mean, raw_std


def save_anomaly_map(anomaly_map: np.ndarray, src_image: Image.Image, dst_path_no_ext: Path):
    width, height = src_image.size
    resized_map_img = Image.fromarray(anomaly_map.astype(np.float32), mode="F").resize((width, height), Image.BICUBIC)
    resized_map = np.asarray(resized_map_img, dtype=np.float32)
    dst_path = dst_path_no_ext.with_suffix(".tiff")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst_path, resized_map)
    return dst_path


def validate_dataset_root(dataset_root: Path, object_names):
    dataset_root = dataset_root.resolve()
    if dataset_root == EVAL_SCRIPT_DIR.resolve():
        dataset_root = dataset_root.parent

    found = [name for name in object_names if (dataset_root / name).is_dir()]
    if not found:
        raise RuntimeError(
            f"{dataset_root} does not look like an MVTec AD dataset root.\n"
            f"Expected category folders such as: {', '.join(object_names[:5])}, ...\n"
            f"Note: {EVAL_SCRIPT_DIR} is only the official evaluation-script directory, not the dataset itself.\n"
            "You need to pass the real dataset root with --dataset-root, for example:\n"
            "  python run_mvtec_ad_benchmark.py --dataset-root /path/to/mvtec\n"
            "and that directory should contain paths like:\n"
            "  /path/to/mvtec/bottle/train/good\n"
            "  /path/to/mvtec/bottle/test/good\n"
            "  /path/to/mvtec/bottle/ground_truth/broken_large"
        )
    return dataset_root, found


def evaluate_results(
    dataset_root: Path,
    anomaly_maps_dir: Path,
    object_names,
    pro_limit: float,
    image_score_index=None,
    classification_score_source: str = "map",
):
    eval_util, parse_dataset_files, calculate_au_pro_au_roc = ensure_eval_importable()
    from pro_curve_util import compute_pro
    from roc_curve_util import compute_classification_roc

    evaluation = {}
    au_pros = []
    au_rocs = []
    p_au_rocs = []
    f1_maxs = []
    segmentation_au_rocs = []
    segmentation_f1_maxs = []

    for object_name in object_names:
        gt_filenames, prediction_filenames = parse_dataset_files(
            object_name=object_name,
            dataset_base_dir=str(dataset_root),
            anomaly_maps_dir=str(anomaly_maps_dir),
        )

        ground_truth, predictions = load_eval_arrays(eval_util, gt_filenames, prediction_filenames)

        pro_curve = compute_pro(
            anomaly_maps=predictions,
            ground_truth_maps=ground_truth,
        )
        au_pro = eval_util.trapezoid(pro_curve[0], pro_curve[1], x_max=pro_limit) / pro_limit
        print(f"AU-PRO (FPR limit: {pro_limit}): {au_pro}")

        binary_labels = [int(np.any(gt > 0)) for gt in ground_truth]
        if classification_score_source == "image_score" and image_score_index is not None:
            image_scores = [
                float(image_score_index.get(str(Path(pred_name)), np.max(pred)))
                for pred_name, pred in zip(prediction_filenames, predictions)
            ]
            roc_curve = compute_classification_roc_from_scores(image_scores, binary_labels)
        else:
            roc_curve = compute_classification_roc(
                anomaly_maps=predictions,
                scoring_function=np.max,
                ground_truth_labels=binary_labels,
            )
            image_scores = [float(np.max(pred)) for pred in predictions]
        au_roc = eval_util.trapezoid(roc_curve[0], roc_curve[1])
        p_au_roc = eval_util.trapezoid(roc_curve[0], roc_curve[1], x_max=pro_limit) / pro_limit

        f1_result = compute_binary_f1_max(image_scores, binary_labels)
        pixel_scores, pixel_labels = flatten_eval_arrays(ground_truth, predictions)
        segmentation_au_roc = compute_binary_auroc(pixel_scores, pixel_labels)
        segmentation_f1_result = compute_binary_f1_max_fast(pixel_scores, pixel_labels)

        print(f"Image-level classification AU-ROC: {au_roc}")
        print(f"Image-level classification pAUROC@FPR<={pro_limit}: {p_au_roc}")
        print(f"Image-level classification F1-max: {f1_result['f1_max']}")
        print(f"Pixel-level segmentation AU-ROC: {segmentation_au_roc}")
        print(f"Pixel-level segmentation F1-max: {segmentation_f1_result['f1_max']}")

        evaluation[object_name] = {
            "au_pro": float(au_pro),
            "segmentation_au_pro": float(au_pro),
            "segmentation_au_roc": float(segmentation_au_roc),
            "segmentation_f1_max": float(segmentation_f1_result["f1_max"]),
            "segmentation_best_threshold": segmentation_f1_result["best_threshold"],
            "segmentation_best_precision": float(segmentation_f1_result["best_precision"]),
            "segmentation_best_recall": float(segmentation_f1_result["best_recall"]),
            "segmentation_num_pixels": int(pixel_labels.size),
            "segmentation_num_anomalous_pixels": int(np.sum(pixel_labels)),
            "segmentation_num_normal_pixels": int(pixel_labels.size - np.sum(pixel_labels)),
            "classification_au_roc": float(au_roc),
            "classification_p_au_roc": float(p_au_roc),
            "classification_f1_max": float(f1_result["f1_max"]),
            "classification_best_threshold": f1_result["best_threshold"],
            "classification_best_precision": float(f1_result["best_precision"]),
            "classification_best_recall": float(f1_result["best_recall"]),
            "classification_num_images": int(len(binary_labels)),
            "classification_num_anomalous": int(np.sum(binary_labels)),
            "classification_num_normal": int(len(binary_labels) - np.sum(binary_labels)),
        }
        au_pros.append(au_pro)
        au_rocs.append(au_roc)
        p_au_rocs.append(p_au_roc)
        f1_maxs.append(f1_result["f1_max"])
        segmentation_au_rocs.append(segmentation_au_roc)
        segmentation_f1_maxs.append(segmentation_f1_result["f1_max"])

    evaluation["mean_au_pro"] = float(np.mean(au_pros))
    evaluation["mean_segmentation_au_pro"] = float(np.mean(au_pros))
    evaluation["mean_segmentation_au_roc"] = float(np.mean(segmentation_au_rocs))
    evaluation["mean_segmentation_f1_max"] = float(np.mean(segmentation_f1_maxs))
    evaluation["mean_classification_au_roc"] = float(np.mean(au_rocs))
    evaluation["mean_classification_p_au_roc"] = float(np.mean(p_au_rocs))
    evaluation["mean_classification_f1_max"] = float(np.mean(f1_maxs))
    evaluation["evaluated_objects"] = object_names
    evaluation["object_names_reference"] = eval_util.OBJECT_NAMES
    return evaluation


def main():
    args = parse_args()
    set_progress_style(args.progress_style)
    if DEVICE != "cuda":
        raise RuntimeError(
            "CUDA GPU is required for this benchmark, but PyTorch did not detect one. "
            "Please install a CUDA-enabled PyTorch build and verify torch.cuda.is_available() is True."
        )

    eval_util, _, _ = ensure_eval_importable()

    dataset_root = args.dataset_root.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    anomaly_maps_dir = output_dir / "anomaly_maps"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if anomaly_maps_dir.exists():
        shutil.rmtree(anomaly_maps_dir)

    dataset_root, available_objects = validate_dataset_root(dataset_root, eval_util.OBJECT_NAMES)
    object_names = args.objects if args.objects else available_objects
    object_names = [name for name in object_names if name in available_objects]
    if not object_names:
        raise RuntimeError("No valid MVTec objects selected for evaluation.")

    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] model_path   = {model_path}")
    print(f"[INFO] output_dir   = {output_dir}")
    print(f"[INFO] device       = {describe_device()}")
    print(f"[INFO] objects      = {', '.join(object_names)}")

    model, tokenizer, image_processor = load_model(model_path)
    attach_mg_refiner(args)
    attach_segad_calibrator(args)

    per_object_runtime = {}
    per_object_thresholds = {}
    benchmark_records = []
    image_score_index = {}

    for object_name in object_names:
        object_root = dataset_root / object_name
        train_good_dir = object_root / "train" / "good"
        test_dir = object_root / "test"

        print(f"\n[INFO] Processing {object_name}")
        activate_object_mg_refiner(args, object_name)
        activate_object_segad_calibrator(args, object_name)
        good_prompts, defect_prompts = get_prompts(object_name)
        good_text_bank = encode_text(good_prompts, tokenizer, model)
        defect_text_bank = encode_text(defect_prompts, tokenizer, model)

        good_proto, good_memory_bank, good_paths = build_good_prototype(train_good_dir, image_processor, model, args)
        threshold, mean_score, std_score, calibration_scores, raw_mean, raw_std = calibrate_threshold(
            good_paths=good_paths,
            image_processor=image_processor,
            model=model,
            good_proto=good_proto,
            good_memory_bank=good_memory_bank,
            good_text_bank=good_text_bank,
            defect_text_bank=defect_text_bank,
            args=args,
        )

        per_object_thresholds[object_name] = {
            "threshold": threshold,
            "train_good_mean": mean_score,
            "train_good_std": std_score,
            "raw_map_mean": raw_mean,
            "raw_map_std": raw_std,
            "good_memory_bank_size": int(good_memory_bank.shape[0]),
            "num_train_good": len(good_paths),
            "calibration_scores": calibration_scores,
        }

        test_image_paths = sorted(test_dir.rglob("*.png"))
        if not test_image_paths:
            raise RuntimeError(f"No test images found in {test_dir}")

        runtimes = []
        for img_path in iter_progress(test_image_paths, desc=f"infer {object_name}"):
            defect_name = img_path.parent.name
            stem = img_path.stem
            image = Image.open(img_path).convert("RGB")

            start_time = time.perf_counter()
            raw_map, norm_map, image_score = compute_maps_and_score(
                image=image,
                image_processor=image_processor,
                model=model,
                good_proto=good_proto,
                good_memory_bank=good_memory_bank,
                good_text_bank=good_text_bank,
                defect_text_bank=defect_text_bank,
                args=args,
            )
            elapsed = time.perf_counter() - start_time
            runtimes.append(elapsed)

            eval_map = standardize_map(raw_map, raw_mean, raw_std)
            map_dst = anomaly_maps_dir / object_name / "test" / defect_name / stem
            saved_map = save_anomaly_map(eval_map, image, map_dst)
            image_score_index[str(map_dst)] = float(image_score)

            benchmark_records.append(
                {
                    "object": object_name,
                    "split": "test",
                    "defect_type": defect_name,
                    "image_path": str(img_path),
                    "anomaly_map_path": str(saved_map),
                    "image_score": float(image_score),
                    "predicted_label": "defect" if image_score >= threshold else "good",
                    "threshold": float(threshold),
                    "raw_map_mean": float(raw_mean),
                    "raw_map_std": float(raw_std),
                    "runtime_sec": float(elapsed),
                }
            )

        per_object_runtime[object_name] = {
            "num_test_images": len(test_image_paths),
            "avg_runtime_sec": float(np.mean(runtimes)),
            "median_runtime_sec": float(np.median(runtimes)),
            "total_runtime_sec": float(np.sum(runtimes)),
        }
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    metrics = evaluate_results(
        dataset_root=dataset_root,
        anomaly_maps_dir=anomaly_maps_dir,
        object_names=object_names,
        pro_limit=args.pro_integration_limit,
        image_score_index=image_score_index,
        classification_score_source=args.classification_score_source,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "device": DEVICE,
        "objects": object_names,
        "resize_short_edge": args.resize_short_edge,
        "max_num_patches": args.max_num_patches,
        "topk_ratio": args.topk_ratio,
        "good_bank_max_patches": args.good_bank_max_patches,
        "good_bank_patches_per_image": args.good_bank_patches_per_image,
        "good_bank_chunk_size": args.good_bank_chunk_size,
        "defect_text_weight": args.defect_text_weight,
        "good_text_weight": args.good_text_weight,
        "proto_distance_weight": args.proto_distance_weight,
        "good_bank_distance_weight": args.good_bank_distance_weight,
        "threshold_std_mult": args.threshold_std_mult,
        "min_threshold_margin": args.min_threshold_margin,
        "pro_integration_limit": args.pro_integration_limit,
        "use_mg": args.use_mg,
        "mg_mask_ratio": args.mg_mask_ratio,
        "mg_mask_dilate_radius": args.mg_mask_dilate_radius,
        "mg_fusion_weight": args.mg_fusion_weight,
        "mg_fusion_mode": args.mg_fusion_mode,
        "mg_mask_mode": args.mg_mask_mode,
        "mg_adaptive_k": args.mg_adaptive_k,
        "mg_adaptive_quantile": args.mg_adaptive_quantile,
        "mg_use_gate": args.mg_use_gate,
        "mg_gate_min_area": args.mg_gate_min_area,
        "mg_gate_max_area": args.mg_gate_max_area,
        "mg_local_only": args.mg_local_only,
        "mg_start_layer": args.mg_start_layer,
        "mg_mask_threshold": args.mg_mask_threshold,
        "mg_neg_bias": args.mg_neg_bias,
        "mg_refiner_checkpoint": str(args.mg_refiner_checkpoint) if args.mg_refiner_checkpoint else None,
        "mg_refiner_dir": str(args.mg_refiner_dir) if args.mg_refiner_dir else None,
        "mg_refiner_threshold": args.mg_refiner_threshold,
        "mg_refiner_mask_mode": args.mg_refiner_mask_mode,
        "mg_refiner_topk_ratio": args.mg_refiner_topk_ratio,
        "mg_refiner_score_ratio": args.mg_refiner_score_ratio,
        "mg_refiner_soft_fusion": args.mg_refiner_soft_fusion,
        "mg_refiner_score_boost": args.mg_refiner_score_boost,
        "mg_refiner_score_power": args.mg_refiner_score_power,
        "mg_use_refiner_weight": args.mg_use_refiner_weight,
        "mg_refiner_fusion_weight": getattr(args, "_mg_refiner_fusion_weight", None),
        "mg_refiner_metadata": getattr(args, "_mg_refiner_metadata", {}),
        "segad_calibrator": str(args.segad_calibrator) if args.segad_calibrator else None,
        "segad_calibrator_dir": str(args.segad_calibrator_dir) if args.segad_calibrator_dir else None,
        "segad_blend_weight": args.segad_blend_weight,
        "segad_fusion_mode": args.segad_fusion_mode,
        "segad_power": args.segad_power,
        "segad_min_confidence": args.segad_min_confidence,
        "image_score_source": args.image_score_source,
        "classification_score_source": args.classification_score_source,
        "segad_calibrator_metadata": getattr(args, "_segad_calibrator_metadata", {}),
        "map_refine_mode": args.map_refine_mode,
        "map_refine_alpha": args.map_refine_alpha,
        "map_refine_bg_sigma": args.map_refine_bg_sigma,
        "map_refine_topk_ratio": args.map_refine_topk_ratio,
        "map_refine_clamp_quantile": args.map_refine_clamp_quantile,
        "thresholds": per_object_thresholds,
        "runtime": per_object_runtime,
        "metrics": metrics,
        "records": benchmark_records,
    }

    metrics_path = metrics_dir / "metrics.json"
    summary_path = output_dir / "benchmark_summary.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[RESULT] Mean AU-PRO               =", f"{metrics['mean_au_pro']:.6f}")
    print("[RESULT] Mean Segmentation AU-ROC  =", f"{metrics['mean_segmentation_au_roc']:.6f}")
    print("[RESULT] Mean Segmentation F1-max  =", f"{metrics['mean_segmentation_f1_max']:.6f}")
    print("[RESULT] Mean Classification AU-ROC =", f"{metrics['mean_classification_au_roc']:.6f}")
    print("[RESULT] Mean Classification pAUROC =", f"{metrics['mean_classification_p_au_roc']:.6f}")
    print("[RESULT] Mean Classification F1-max =", f"{metrics['mean_classification_f1_max']:.6f}")
    print("[RESULT] Metrics JSON              =", metrics_path)
    print("[RESULT] Benchmark Summary         =", summary_path)


if __name__ == "__main__":
    main()
