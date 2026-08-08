import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from visa_evaluation import compute_classification_roc, compute_pro, trapezoid

from map_refinement import (
    MAP_REFINE_ALPHA,
    MAP_REFINE_BG_SIGMA,
    MAP_REFINE_CLAMP_QUANTILE,
    MAP_REFINE_MODE,
    MAP_REFINE_TOPK_RATIO,
)
from run_mvtec_ad_benchmark import (
    DEVICE,
    REPO_ROOT,
    MODEL_NAME_OR_PATH,
    PRO_INTEGRATION_LIMIT,
    activate_object_mg_refiner,
    activate_object_segad_calibrator,
    attach_segad_calibrator,
    attach_mg_refiner,
    build_good_prototype,
    calibrate_threshold,
    compute_binary_auroc,
    compute_binary_f1_max,
    compute_binary_f1_max_fast,
    compute_classification_roc_from_scores,
    compute_maps_and_score,
    describe_device,
    encode_text,
    get_prompts as get_mvtec_prompts,
    load_model,
    set_progress_style,
    standardize_map,
    save_anomaly_map,
)


DEFAULT_DATASET_ROOT = REPO_ROOT / "visa"
DEFAULT_SPLIT_CSV = DEFAULT_DATASET_ROOT / "split_csv" / "1cls.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "visa_fgclip2_benchmark"

VISA_PROMPTS = {
    "candle": {
        "good": ["a normal candle", "an intact candle", "a candle without defects"],
        "defect": [
            "a candle with melted wax",
            "a candle with foreign particles",
            "a candle with extra wax",
            "a candle with a chunk of wax missing",
            "a candle with abnormal candle wick",
            "a candle with damaged packaging corner",
            "a candle with a different color spot",
        ],
    },
    "capsules": {
        "good": ["normal capsules", "intact capsules", "capsules without defects"],
        "defect": [
            "capsules with scratch",
            "capsules with discoloration",
            "capsules with misshape",
            "capsules with leak",
            "capsules with bubble",
        ],
    },
    "cashew": {
        "good": ["a normal cashew", "an intact cashew", "a clean cashew without defects"],
        "defect": [
            "a cashew with breakage",
            "a cashew with small scratches",
            "a cashew with burnt area",
            "a cashew stuck together",
            "a cashew with spot defect",
        ],
    },
    "chewinggum": {
        "good": ["normal chewing gum", "intact chewing gum", "chewing gum without defects"],
        "defect": [
            "chewing gum with corner missing",
            "chewing gum with scratches",
            "chewing gum with a chunk missing",
            "chewing gum with color spot",
            "chewing gum with cracks",
        ],
    },
    "fryum": {
        "good": ["a normal fryum", "an intact fryum", "a fryum without defects"],
        "defect": [
            "a fryum with breakage",
            "a fryum with scratches",
            "a fryum with burnt area",
            "a fryum with color spot",
            "fryum stuck together",
        ],
    },
    "macaroni1": {
        "good": ["normal macaroni", "intact macaroni", "macaroni without defects"],
        "defect": [
            "macaroni with color spot",
            "macaroni with small chip around edge",
            "macaroni with small scratches",
            "macaroni with breakage",
            "macaroni with cracks",
        ],
    },
    "macaroni2": {
        "good": ["normal macaroni", "intact macaroni", "macaroni without defects"],
        "defect": [
            "macaroni with color spot",
            "macaroni with small chip around edge",
            "macaroni with small scratches",
            "macaroni with breakage",
            "macaroni with cracks",
        ],
    },
    "pcb1": {
        "good": ["a normal circuit board", "an intact pcb", "a pcb without defects"],
        "defect": [
            "a pcb with bent component",
            "a pcb with scratch",
            "a pcb with missing component",
            "a pcb with melt defect",
        ],
    },
    "pcb2": {
        "good": ["a normal circuit board", "an intact pcb", "a pcb without defects"],
        "defect": [
            "a pcb with bent component",
            "a pcb with scratch",
            "a pcb with missing component",
            "a pcb with melt defect",
        ],
    },
    "pcb3": {
        "good": ["a normal circuit board", "an intact pcb", "a pcb without defects"],
        "defect": [
            "a pcb with bent component",
            "a pcb with scratch",
            "a pcb with missing component",
            "a pcb with melt defect",
        ],
    },
    "pcb4": {
        "good": ["a normal circuit board", "an intact pcb", "a pcb without defects"],
        "defect": [
            "a pcb with scratch",
            "a pcb with extra component",
            "a pcb with missing component",
            "a pcb with wrong placement",
            "a pcb with damage",
            "a pcb with burnt area",
            "a pcb with dirt",
        ],
    },
    "pipe_fryum": {
        "good": ["a normal pipe fryum", "an intact pipe fryum", "a pipe fryum without defects"],
        "defect": [
            "a pipe fryum with breakage",
            "a pipe fryum with small scratches",
            "a pipe fryum with burnt area",
            "pipe fryum stuck together",
            "a pipe fryum with color spot",
            "a pipe fryum with cracks",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FG-CLIP on the VisA dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Path to the VisA dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
        help=f"Path to the VisA split csv file. Default: {DEFAULT_SPLIT_CSV}",
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
        help="Subset of VisA categories to evaluate. Default: all categories from the split csv.",
    )
    parser.add_argument("--resize-short-edge", type=int, default=1024)
    parser.add_argument("--max-num-patches", type=int, default=4096)
    parser.add_argument("--topk-ratio", type=float, default=0.002)
    parser.add_argument("--good-bank-max-patches", type=int, default=50000)
    parser.add_argument("--good-bank-patches-per-image", type=int, default=256)
    parser.add_argument("--good-bank-chunk-size", type=int, default=8192)
    parser.add_argument("--defect-text-weight", type=float, default=1.0)
    parser.add_argument("--good-text-weight", type=float, default=0.15)
    parser.add_argument("--proto-distance-weight", type=float, default=0.35)
    parser.add_argument("--good-bank-distance-weight", type=float, default=0.85)
    parser.add_argument("--threshold-std-mult", type=float, default=3.0)
    parser.add_argument("--min-threshold-margin", type=float, default=0.015)
    parser.add_argument("--pro-integration-limit", type=float, default=PRO_INTEGRATION_LIMIT)
    parser.add_argument(
        "--use-mg",
        action="store_true",
        help="Enable two-stage MG patch-mask refinement for dense image features.",
    )
    parser.add_argument("--mg-mask-ratio", type=float, default=0.10)
    parser.add_argument("--mg-mask-dilate-radius", type=int, default=1)
    parser.add_argument("--mg-fusion-weight", type=float, default=0.10)
    parser.add_argument(
        "--mg-fusion-mode",
        choices=["none", "linear", "residual", "positive"],
        default="positive",
    )
    parser.add_argument(
        "--mg-mask-mode",
        choices=["ratio", "adaptive"],
        default="ratio",
        help="MG mask generation mode: top-k ratio or adaptive score threshold.",
    )
    parser.add_argument("--mg-adaptive-k", type=float, default=1.0)
    parser.add_argument("--mg-adaptive-quantile", type=float, default=0.90)
    parser.add_argument("--mg-use-gate", action="store_true")
    parser.add_argument("--mg-gate-min-area", type=float, default=0.001)
    parser.add_argument("--mg-gate-max-area", type=float, default=0.3)
    parser.add_argument(
        "--mg-local-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply MG residual only inside the selected MG mask region.",
    )
    parser.add_argument("--mg-start-layer", type=int, default=3)
    parser.add_argument("--mg-mask-threshold", type=float, default=0.5)
    parser.add_argument("--mg-neg-bias", type=float, default=-1e4)
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
    parser.add_argument("--mg-refiner-threshold", type=float, default=0.5)
    parser.add_argument(
        "--mg-refiner-mask-mode",
        choices=["threshold", "topk", "score_intersect"],
        default="score_intersect",
        help="How the trained MG refiner selects patches.",
    )
    parser.add_argument("--mg-refiner-topk-ratio", type=float, default=0.05)
    parser.add_argument("--mg-refiner-score-ratio", type=float, default=0.05)
    parser.add_argument(
        "--mg-refiner-soft-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use refiner probabilities as a soft local gate for the positive residual.",
    )
    parser.add_argument("--mg-refiner-score-boost", type=float, default=0.0)
    parser.add_argument("--mg-refiner-score-power", type=float, default=1.0)
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
    parser.add_argument("--segad-blend-weight", type=float, default=0.0)
    parser.add_argument("--segad-fusion-mode", choices=["linear", "positive"], default="positive")
    parser.add_argument("--segad-power", type=float, default=1.0)
    parser.add_argument("--segad-min-confidence", type=float, default=0.55)
    parser.add_argument("--image-score-source", choices=["final", "pre_segad"], default="final")
    parser.add_argument("--classification-score-source", choices=["map", "image_score"], default="map")
    parser.add_argument(
        "--map-refine-mode",
        choices=["none", "local_contrast", "topk_contrast"],
        default=MAP_REFINE_MODE,
    )
    parser.add_argument("--map-refine-alpha", type=float, default=MAP_REFINE_ALPHA)
    parser.add_argument("--map-refine-bg-sigma", type=float, default=MAP_REFINE_BG_SIGMA)
    parser.add_argument("--map-refine-topk-ratio", type=float, default=MAP_REFINE_TOPK_RATIO)
    parser.add_argument("--map-refine-clamp-quantile", type=float, default=MAP_REFINE_CLAMP_QUANTILE)
    parser.add_argument("--progress_style", choices=["stage", "live"], default="stage")
    return parser.parse_args()


def load_visa_split(split_csv: Path, dataset_root: Path):
    if not split_csv.exists():
        raise RuntimeError(f"VisA split csv not found: {split_csv}")

    entries = {}
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_name = row["object"]
            split_name = row["split"]
            item = {
                "label": row["label"],
                "image_path": (dataset_root / row["image"]).resolve(),
                "mask_path": (dataset_root / row["mask"]).resolve() if row["mask"] else None,
            }
            entries.setdefault(object_name, {}).setdefault(split_name, []).append(item)
    return entries


def validate_visa_root(dataset_root: Path, split_entries):
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise RuntimeError(f"VisA dataset root does not exist: {dataset_root}")

    available = []
    for object_name, object_entries in split_entries.items():
        train_items = object_entries.get("train", [])
        test_items = object_entries.get("test", [])
        if train_items and test_items:
            available.append(object_name)

    if not available:
        raise RuntimeError(
            f"{dataset_root} does not look like a valid VisA root. "
            "Expected category folders such as candle/, capsules/, pcb1/, ... and a valid split csv."
        )
    return dataset_root, sorted(available)


def get_prompts(object_name: str):
    if object_name in VISA_PROMPTS:
        prompt_cfg = VISA_PROMPTS[object_name]
        return prompt_cfg["good"], prompt_cfg["defect"]
    return get_mvtec_prompts(object_name)


def load_mask(mask_path: Path, image_size):
    if mask_path is None:
        width, height = image_size
        return np.zeros((height, width), dtype=np.uint8)

    mask = Image.open(mask_path)
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def resize_eval_map_to_image(anomaly_map: np.ndarray, image_size):
    resized = Image.fromarray(anomaly_map.astype(np.float32), mode="F").resize(image_size, Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32)


def evaluate_object(
    predictions,
    ground_truth,
    image_scores,
    image_labels,
    pro_limit: float,
    classification_score_source: str = "map",
):
    pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=ground_truth)
    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=pro_limit) / pro_limit

    if classification_score_source == "image_score":
        roc_curve = compute_classification_roc_from_scores(image_scores, image_labels)
    else:
        roc_curve = compute_classification_roc(
            anomaly_maps=predictions,
            scoring_function=np.max,
            ground_truth_labels=image_labels,
        )
    classification_au_roc = trapezoid(roc_curve[0], roc_curve[1])
    classification_p_au_roc = trapezoid(roc_curve[0], roc_curve[1], x_max=pro_limit) / pro_limit
    classification_f1 = compute_binary_f1_max(image_scores, image_labels)

    pixel_scores = np.concatenate([pred.reshape(-1) for pred in predictions]).astype(np.float32, copy=False)
    pixel_labels = np.concatenate([gt.reshape(-1) for gt in ground_truth]).astype(np.uint8, copy=False)
    segmentation_au_roc = compute_binary_auroc(pixel_scores, pixel_labels)
    segmentation_f1 = compute_binary_f1_max_fast(pixel_scores, pixel_labels)

    return {
        "au_pro": float(au_pro),
        "segmentation_au_pro": float(au_pro),
        "segmentation_au_roc": float(segmentation_au_roc),
        "segmentation_f1_max": float(segmentation_f1["f1_max"]),
        "segmentation_best_threshold": segmentation_f1["best_threshold"],
        "segmentation_best_precision": float(segmentation_f1["best_precision"]),
        "segmentation_best_recall": float(segmentation_f1["best_recall"]),
        "segmentation_num_pixels": int(pixel_labels.size),
        "segmentation_num_anomalous_pixels": int(np.sum(pixel_labels)),
        "segmentation_num_normal_pixels": int(pixel_labels.size - np.sum(pixel_labels)),
        "classification_au_roc": float(classification_au_roc),
        "classification_p_au_roc": float(classification_p_au_roc),
        "classification_f1_max": float(classification_f1["f1_max"]),
        "classification_best_threshold": classification_f1["best_threshold"],
        "classification_best_precision": float(classification_f1["best_precision"]),
        "classification_best_recall": float(classification_f1["best_recall"]),
        "classification_num_images": int(len(image_labels)),
        "classification_num_anomalous": int(np.sum(image_labels)),
        "classification_num_normal": int(len(image_labels) - np.sum(image_labels)),
    }


def main():
    args = parse_args()
    set_progress_style(args.progress_style)
    if DEVICE != "cuda":
        raise RuntimeError(
            "CUDA GPU is required for this benchmark, but PyTorch did not detect one. "
            "Please install a CUDA-enabled PyTorch build and verify torch.cuda.is_available() is True."
        )

    dataset_root = args.dataset_root.resolve()
    split_csv = args.split_csv.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    anomaly_maps_dir = output_dir / "anomaly_maps"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    split_entries = load_visa_split(split_csv, dataset_root)
    dataset_root, available_objects = validate_visa_root(dataset_root, split_entries)
    object_names = args.objects if args.objects else available_objects
    object_names = [name for name in object_names if name in available_objects]
    if not object_names:
        raise RuntimeError("No valid VisA objects selected for evaluation.")

    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] split_csv    = {split_csv}")
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
    metrics = {}

    mean_au_pro = []
    mean_seg_au_roc = []
    mean_seg_f1 = []
    mean_cls_au_roc = []
    mean_cls_p_au_roc = []
    mean_cls_f1 = []

    for object_name in object_names:
        object_entries = split_entries[object_name]
        train_entries = [item for item in object_entries["train"] if item["label"] == "normal"]
        test_entries = object_entries["test"]
        if not train_entries or not test_entries:
            raise RuntimeError(f"VisA object {object_name} is missing train/test entries in {split_csv}")

        print(f"\n[INFO] Processing {object_name}")
        activate_object_mg_refiner(args, object_name)
        activate_object_segad_calibrator(args, object_name)
        good_prompts, defect_prompts = get_prompts(object_name)
        good_text_bank = encode_text(good_prompts, tokenizer, model)
        defect_text_bank = encode_text(defect_prompts, tokenizer, model)

        good_dir = train_entries[0]["image_path"].parents[2]
        good_proto, good_memory_bank, good_paths = build_good_prototype(good_dir, image_processor, model, args)
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

        runtimes = []
        predictions = []
        ground_truth = []
        image_scores = []
        image_labels = []

        for item in test_entries:
            image = Image.open(item["image_path"]).convert("RGB")
            start_time = time.perf_counter()
            raw_map, _, image_score = compute_maps_and_score(
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
            resized_eval_map = resize_eval_map_to_image(eval_map, image.size)
            predictions.append(resized_eval_map)

            gt_mask = load_mask(item["mask_path"], image.size)
            ground_truth.append(gt_mask)
            image_labels.append(int(item["label"] != "normal"))
            image_scores.append(float(image_score))

            defect_name = item["label"]
            stem = item["image_path"].stem
            map_dst = anomaly_maps_dir / object_name / "test" / defect_name / stem
            saved_map = save_anomaly_map(eval_map, image, map_dst)

            benchmark_records.append(
                {
                    "object": object_name,
                    "split": "test",
                    "label": item["label"],
                    "image_path": str(item["image_path"]),
                    "mask_path": str(item["mask_path"]) if item["mask_path"] is not None else None,
                    "anomaly_map_path": str(saved_map),
                    "image_score": float(image_score),
                    "predicted_label": "anomaly" if image_score >= threshold else "normal",
                    "threshold": float(threshold),
                    "raw_map_mean": float(raw_mean),
                    "raw_map_std": float(raw_std),
                    "runtime_sec": float(elapsed),
                }
            )

        per_object_runtime[object_name] = {
            "num_test_images": len(test_entries),
            "avg_runtime_sec": float(np.mean(runtimes)),
            "median_runtime_sec": float(np.median(runtimes)),
            "total_runtime_sec": float(np.sum(runtimes)),
        }

        object_metrics = evaluate_object(
            predictions,
            ground_truth,
            image_scores,
            image_labels,
            args.pro_integration_limit,
            args.classification_score_source,
        )
        metrics[object_name] = object_metrics

        mean_au_pro.append(object_metrics["au_pro"])
        mean_seg_au_roc.append(object_metrics["segmentation_au_roc"])
        mean_seg_f1.append(object_metrics["segmentation_f1_max"])
        mean_cls_au_roc.append(object_metrics["classification_au_roc"])
        mean_cls_p_au_roc.append(object_metrics["classification_p_au_roc"])
        mean_cls_f1.append(object_metrics["classification_f1_max"])

        print(f"AU-PRO (FPR limit: {args.pro_integration_limit}): {object_metrics['au_pro']}")
        print(f"Pixel-level segmentation AU-ROC: {object_metrics['segmentation_au_roc']}")
        print(f"Pixel-level segmentation F1-max: {object_metrics['segmentation_f1_max']}")
        print(f"Image-level classification AU-ROC: {object_metrics['classification_au_roc']}")
        print(f"Image-level classification pAUROC@FPR<={args.pro_integration_limit}: {object_metrics['classification_p_au_roc']}")
        print(f"Image-level classification F1-max: {object_metrics['classification_f1_max']}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    metrics["mean_au_pro"] = float(np.mean(mean_au_pro))
    metrics["mean_segmentation_au_pro"] = float(np.mean(mean_au_pro))
    metrics["mean_segmentation_au_roc"] = float(np.mean(mean_seg_au_roc))
    metrics["mean_segmentation_f1_max"] = float(np.mean(mean_seg_f1))
    metrics["mean_classification_au_roc"] = float(np.mean(mean_cls_au_roc))
    metrics["mean_classification_p_au_roc"] = float(np.mean(mean_cls_p_au_roc))
    metrics["mean_classification_f1_max"] = float(np.mean(mean_cls_f1))
    metrics["evaluated_objects"] = object_names

    summary = {
        "dataset_root": str(dataset_root),
        "split_csv": str(split_csv),
        "model_path": str(model_path),
        "device": describe_device(),
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
