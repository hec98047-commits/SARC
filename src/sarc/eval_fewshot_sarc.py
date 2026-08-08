from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch

from lec_reference import (
    collect_normal_reference_patches,
    compute_normal_reference_distance,
    fuse_pgcre_with_nrd,
    normalize_map,
    resize_distance_to_heatmap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SARC few-shot evaluation with SP, ARC, and LEC.")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--sarc_model_path", dest="mg_model_path", type=Path, required=True)
    parser.add_argument("--sampled_normals", "--sampled_normal_paths", dest="sampled_normals", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--subset", choices=["all", "small", "tiny"], default="all")
    parser.add_argument("--classes", default=None, help="Optional comma-separated class list for quick per-class debugging.")
    parser.add_argument("--tile_mode", default="2x2")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--lambda_weight", type=float, default=0.5)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument(
        "--mg_mask_ratio",
        type=float,
        default=0.2,
        help="Top candidate ratio; 0.2 is equivalent to q=0.8.",
    )
    parser.add_argument("--enable_fg", action="store_true")
    parser.add_argument("--enable_positive_fusion", action="store_true")
    parser.add_argument(
        "--disable_arc",
        dest="disable_mg_branch",
        action="store_true",
        help="Disable the ARC stage.",
    )
    parser.add_argument(
        "--disable_lec",
        dest="disable_ms_branch",
        action="store_true",
        help="Disable the LEC stage.",
    )
    parser.add_argument(
        "--mg_fusion_mode",
        choices=["positive", "direct", "off"],
        default="direct",
    )
    parser.add_argument("--mg_fusion_weight", type=float, default=0.1)
    parser.add_argument("--prompt_style", default="spatial_aware")
    parser.add_argument("--adaptive_prompt_policy_path", type=Path, default=None)
    parser.add_argument("--aupro_segauc_only", action="store_true")
    parser.add_argument("--enable_dual_layer_guidance", action="store_true")
    parser.add_argument("--local_feature_layer", type=int, default=5)
    parser.add_argument("--enable_nrs", action="store_true")
    parser.add_argument("--nrs_mode", choices=["subtractive", "multiplicative"], default="subtractive")
    parser.add_argument("--nrs_alpha", type=float, default=0.2)
    parser.add_argument("--nrs_gamma", type=float, default=1.0)
    parser.add_argument("--nrs_tau", type=float, default=0.0)
    parser.add_argument("--nrs_power", type=float, default=1.0)
    parser.add_argument(
        "--normal_reference_calibration",
        dest="enable_nrd",
        action="store_true",
    )
    parser.add_argument("--nrd_beta", type=float, default=0.3)
    parser.add_argument("--nrd_fusion_mode", choices=["add", "residual", "multiply", "candidate_pgcre_rank", "rank_residual", "agreement_gated_add"], default="add")
    parser.add_argument("--nrd_feature_layer", type=int, default=5)
    parser.add_argument("--nrd_max_ref_patches", type=int, default=4096)
    parser.add_argument(
        "--reference_selection",
        dest="nrd_ref_selection",
        choices=["uniform", "fps"],
        default="uniform",
    )
    parser.add_argument("--nrd_zscore", choices=["image", "class"], default="image")
    parser.add_argument("--nrd_eps", type=float, default=1e-6)
    parser.add_argument("--nrd_pro_q", type=float, default=0.85)
    parser.add_argument("--nrd_pro_beta", type=float, default=10.0)
    parser.add_argument("--nrd_eta_nrd", type=float, default=0.10)
    parser.add_argument("--nrd_lambda_prior", type=float, default=0.03)
    parser.add_argument("--nrd_agree_tau", type=float, default=0.20)
    parser.add_argument("--nrd_class_route_json", type=Path, default=None)
    parser.add_argument("--nrd_use_fgclip_patch_features", action="store_true")
    parser.add_argument("--nrd_only", action="store_true")
    parser.add_argument("--low_storage", default="true")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--arc_checkpoint", dest="token_modulator_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--fp_metrics_only",
        action="store_true",
        help=(
            "Skip AU-PRO/AUROC aggregation and report normal-region false-positive "
            "statistics after per-image min-max normalization."
        ),
    )
    parser.add_argument(
        "--fp_thresholds",
        default="0.5,0.6,0.7",
        help="Comma-separated thresholds used by --fp_metrics_only.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_fp_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"FP threshold must be in [0, 1], got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one FP threshold is required.")
    return values


def _threshold_key(value: float) -> str:
    return f"t{int(round(value * 100)):02d}"


def _normalize_per_image(anomaly_map: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(anomaly_map, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def _compute_fp_metrics(
    records: list[dict],
    method: str,
    thresholds: list[float],
) -> dict:
    row: dict[str, int | float] = {
        "num_images": len(records),
        "num_normal_images": 0,
        "num_anomaly_images": 0,
        "bg_pixels": 0,
        "normal_image_bg_pixels": 0,
        "anomaly_image_bg_pixels": 0,
        "bg_score_sum": 0.0,
        "bg_p95_image_sum": 0.0,
    }
    for threshold in thresholds:
        key = _threshold_key(threshold)
        row[f"fp_bg_count_{key}"] = 0
        row[f"fp_normal_image_count_{key}"] = 0
        row[f"fp_anomaly_bg_count_{key}"] = 0

    for item in records:
        normalized = _normalize_per_image(item[method]["map"])
        mask = np.asarray(item["mask"], dtype=np.uint8)
        background = mask == 0
        if normalized.shape != background.shape:
            raise ValueError(
                f"FP metric shape mismatch: prediction={normalized.shape}, mask={background.shape}"
            )
        background_scores = normalized[background]
        background_count = int(background_scores.size)
        if background_count == 0:
            continue
        is_anomaly = bool(item["label"])
        row["num_anomaly_images" if is_anomaly else "num_normal_images"] += 1
        row["bg_pixels"] += background_count
        target_pixel_key = (
            "anomaly_image_bg_pixels" if is_anomaly else "normal_image_bg_pixels"
        )
        row[target_pixel_key] += background_count
        row["bg_score_sum"] += float(np.sum(background_scores, dtype=np.float64))
        row["bg_p95_image_sum"] += float(np.quantile(background_scores, 0.95))

        for threshold in thresholds:
            key = _threshold_key(threshold)
            activated = int(np.count_nonzero(background_scores >= threshold))
            row[f"fp_bg_count_{key}"] += activated
            if is_anomaly:
                row[f"fp_anomaly_bg_count_{key}"] += activated
            else:
                row[f"fp_normal_image_count_{key}"] += activated

    bg_pixels = max(int(row["bg_pixels"]), 1)
    num_images = max(int(row["num_images"]), 1)
    row["bg_mean_score"] = float(row["bg_score_sum"]) / bg_pixels
    row["bg_p95_image_mean"] = float(row["bg_p95_image_sum"]) / num_images
    normal_pixels = max(int(row["normal_image_bg_pixels"]), 1)
    anomaly_pixels = max(int(row["anomaly_image_bg_pixels"]), 1)
    for threshold in thresholds:
        key = _threshold_key(threshold)
        row[f"fp_bg_rate_{key}"] = float(row[f"fp_bg_count_{key}"]) / bg_pixels
        row[f"fp_normal_image_rate_{key}"] = (
            float(row[f"fp_normal_image_count_{key}"]) / normal_pixels
        )
        row[f"fp_anomaly_bg_rate_{key}"] = (
            float(row[f"fp_anomaly_bg_count_{key}"]) / anomaly_pixels
        )
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No FP metric rows were produced.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_fp_rows(rows: list[dict], thresholds: list[float]) -> dict:
    metric_fields = ["bg_mean_score", "bg_p95_image_mean"]
    for threshold in thresholds:
        key = _threshold_key(threshold)
        metric_fields.extend(
            [
                f"fp_bg_rate_{key}",
                f"fp_normal_image_rate_{key}",
                f"fp_anomaly_bg_rate_{key}",
            ]
        )
    macro = {
        field: float(np.mean([float(row[field]) for row in rows]))
        for field in metric_fields
    }

    pooled: dict[str, float | int] = {
        "num_classes": len(rows),
        "num_images": int(sum(int(row["num_images"]) for row in rows)),
        "num_normal_images": int(sum(int(row["num_normal_images"]) for row in rows)),
        "num_anomaly_images": int(sum(int(row["num_anomaly_images"]) for row in rows)),
        "bg_pixels": int(sum(int(row["bg_pixels"]) for row in rows)),
        "normal_image_bg_pixels": int(
            sum(int(row["normal_image_bg_pixels"]) for row in rows)
        ),
        "anomaly_image_bg_pixels": int(
            sum(int(row["anomaly_image_bg_pixels"]) for row in rows)
        ),
    }
    pooled["bg_mean_score"] = float(
        sum(float(row["bg_score_sum"]) for row in rows)
    ) / max(int(pooled["bg_pixels"]), 1)
    pooled["bg_p95_image_mean"] = float(
        sum(float(row["bg_p95_image_sum"]) for row in rows)
    ) / max(int(pooled["num_images"]), 1)
    for threshold in thresholds:
        key = _threshold_key(threshold)
        pooled[f"fp_bg_rate_{key}"] = float(
            sum(int(row[f"fp_bg_count_{key}"]) for row in rows)
        ) / max(int(pooled["bg_pixels"]), 1)
        pooled[f"fp_normal_image_rate_{key}"] = float(
            sum(int(row[f"fp_normal_image_count_{key}"]) for row in rows)
        ) / max(int(pooled["normal_image_bg_pixels"]), 1)
        pooled[f"fp_anomaly_bg_rate_{key}"] = float(
            sum(int(row[f"fp_anomaly_bg_count_{key}"]) for row in rows)
        ) / max(int(pooled["anomaly_image_bg_pixels"]), 1)
    return {
        "normalization": "per-image min-max",
        "thresholds": thresholds,
        "macro_class_mean": macro,
        "pooled_pixels": pooled,
    }


def _load_optional_class_route(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = _load_json(path)
    route = payload.get("classes", payload)
    out: dict[str, str] = {}
    for key, value in route.items():
        out[str(key)] = str(value).strip().lower()
    return out


def main():
    args = parse_args()
    module_a_enabled = args.prompt_style != "default" or args.adaptive_prompt_policy_path is not None
    module_b_enabled = not args.disable_mg_branch
    module_c_enabled = not args.disable_ms_branch
    print(f"[SP] {'enabled' if module_a_enabled else 'disabled'}")
    print(f"[ARC] {'enabled' if module_b_enabled else 'disabled'}")
    print(f"[LEC] {'enabled' if module_c_enabled else 'disabled'}")
    class_route = _load_optional_class_route(args.nrd_class_route_json)
    sampled = _load_json(args.sampled_normals)
    class_to_paths = {
        class_name: [Path(p) for p in paths]
        for class_name, paths in sampled.get("classes", {}).items()
    }
    requested_classes = None
    if args.classes:
        requested_classes = [item.strip() for item in str(args.classes).split(",") if item.strip()]
        missing = [item for item in requested_classes if item not in class_to_paths]
        if missing:
            raise RuntimeError(f"Requested classes not found in sampled normals: {missing}")
    image_path_to_class = {
        str(Path(path).resolve()).lower(): class_name
        for class_name, paths in class_to_paths.items()
        for path in paths
    }

    import sarc_runtime as pgcre
    import run_mvtec_ad_benchmark as mvbench

    original_pgcre_build = pgcre.build_good_prototype
    original_mv_build = mvbench.build_good_prototype
    original_infer_ms_fb_mg = pgcre.infer_ms_fb_mg
    original_image_open = pgcre.Image.open
    original_pgcre_load_model = pgcre.load_model
    original_compute_subset_metrics = pgcre.compute_subset_metrics
    nrs_memory_by_class: dict[str, torch.Tensor] = {}
    nrd_memory_by_class: dict[str, torch.Tensor] = {}
    active_reference_class = {"value": None}
    last_opened_class_name = {"value": None}
    fp_thresholds = _parse_fp_thresholds(args.fp_thresholds)
    fp_rows: list[dict] = []

    def load_model_with_token_modulator(model_path):
        model, tokenizer, processor = original_pgcre_load_model(model_path)
        if (
            args.token_modulator_checkpoint is not None
            and Path(model_path).resolve() == args.mg_model_path.resolve()
        ):
            payload = torch.load(
                args.token_modulator_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            model.vision_model.encoder.token_modulator.load_state_dict(payload["adapter"], strict=True)
            model.vision_model.encoder.token_modulation_enabled = True
            print("[ARC] checkpoint loaded")
        return model, tokenizer, processor

    def topk_score_from_map(anomaly_map: np.ndarray, topk_ratio: float = 0.05) -> float:
        flat = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return 0.0
        keep = max(1, int(math.ceil(flat.size * max(float(topk_ratio), 1e-6))))
        return float(np.mean(np.partition(flat, -keep)[-keep:]))

    def open_with_class_tag(path, *open_args, **open_kwargs):
        image = original_image_open(path, *open_args, **open_kwargs)
        resolved = str(Path(path).resolve()).lower()
        class_name = image_path_to_class.get(resolved)
        if class_name is None:
            normalized = resolved.replace("/", "\\")
            for known_class in class_to_paths:
                if f"\\{known_class.lower()}\\" in normalized:
                    class_name = known_class
                    break
        if class_name is not None:
            setattr(image, "_fewshot_class_name", class_name)
            last_opened_class_name["value"] = class_name
        return image

    def infer_class_from_image(image) -> str | None:
        class_name = getattr(image, "_fewshot_class_name", None)
        if class_name:
            return str(class_name)
        filename = getattr(image, "filename", None)
        if filename:
            resolved = str(Path(filename).resolve()).lower()
            for known_class in class_to_paths:
                if f"\\{known_class.lower()}\\" in resolved.replace("/", "\\"):
                    return known_class
        return last_opened_class_name.get("value")

    def infer_class_name_from_good_dir(good_dir: Path) -> str | None:
        parts = {part.lower(): part for part in Path(good_dir).parts}
        for class_name in class_to_paths:
            if class_name.lower() in parts:
                return class_name
        # Common MVTec/VisA folder layout: <class>/train/good.
        parents = Path(good_dir).parents
        if len(parents) > 1 and parents[1].name in class_to_paths:
            return parents[1].name
        return None

    @torch.inference_mode()
    def build_good_prototype_from_sampled(good_dir: Path, image_processor, model, runtime_args):
        class_name = infer_class_name_from_good_dir(good_dir)
        if class_name is None:
            raise RuntimeError(
                f"Could not infer class name from good_dir={good_dir}. "
                "Refusing to fall back to full train/good under few-shot protocol."
            )
        good_paths = class_to_paths.get(class_name, [])
        if not good_paths:
            raise RuntimeError(
                f"No sampled normals for {class_name} in {args.sampled_normals}. "
                "Refusing to fall back to full train/good under few-shot protocol."
            )
        if active_reference_class["value"] != class_name:
            nrs_memory_by_class.clear()
            nrd_memory_by_class.clear()
            active_reference_class["value"] = class_name
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[FewShot] using {len(good_paths)} sampled normal image(s) for {class_name}")
        good_vecs = []
        memory_tokens = []
        for img_path in pgcre.iter_progress(good_paths, desc=f"fewshot prototype {class_name}"):
            image = Image.open(img_path).convert("RGB")
            dense_feat, _, _, _ = pgcre.encode_dense_image(
                image=image,
                image_processor=image_processor,
                model=model,
                resize_target=runtime_args.resize_short_edge,
                max_num_patches=runtime_args.max_num_patches,
                feature_layer=getattr(runtime_args, "feature_layer", 0),
            )
            img_vec = mvbench.l2norm(dense_feat.mean(dim=0), dim=-1)
            good_vecs.append(img_vec)
            sampled_tokens = mvbench.sample_good_memory_tokens(dense_feat, runtime_args.good_bank_patches_per_image)
            memory_tokens.append(sampled_tokens)
        if args.enable_nrs and int(getattr(runtime_args, "feature_layer", 0) or 0) == 0:
            nrs_memory = torch.cat(memory_tokens, dim=0).detach().float()
            nrs_memory = nrs_memory / nrs_memory.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            nrs_memory_by_class[class_name] = nrs_memory
        if args.enable_nrd and int(getattr(runtime_args, "feature_layer", 0) or 0) == 0 and class_name not in nrd_memory_by_class:
            nrd_model = model if args.nrd_use_fgclip_patch_features else model
            nrd_processor = image_processor
            nrd_memory_by_class[class_name] = collect_normal_reference_patches(
                model=nrd_model,
                normal_image_paths=good_paths,
                preprocess=nrd_processor,
                device=good_vecs[0].device if good_vecs else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                feature_layer=int(args.nrd_feature_layer),
                max_ref_patches=int(args.nrd_max_ref_patches),
                use_fp16=True,
                resize_short_edge=int(runtime_args.resize_short_edge),
                max_num_patches=int(runtime_args.max_num_patches),
                encode_dense_image_fn=pgcre.encode_dense_image,
                selection_mode=str(args.nrd_ref_selection),
            )
        if not good_vecs:
            raise RuntimeError(f"No sampled normal images available for {class_name}.")
        good_proto = mvbench.l2norm(torch.stack(good_vecs, dim=0).mean(dim=0), dim=-1)
        good_memory_bank = mvbench.finalize_good_memory_bank(memory_tokens, runtime_args.good_bank_max_patches)
        return good_proto, good_memory_bank, good_paths

    @torch.inference_mode()
    def infer_ms_fb_mg_with_nrs(image, baseline_ctx, mg_ctx, baseline_stats, mg_stats, runtime_args, **kwargs):
        outputs = original_infer_ms_fb_mg(
            image,
            baseline_ctx,
            mg_ctx,
            baseline_stats,
            mg_stats,
            runtime_args,
            **kwargs,
        )
        if not args.enable_nrs and not args.enable_nrd:
            return outputs
        class_name = infer_class_from_image(image)
        has_nrs_memory = class_name is not None and class_name in nrs_memory_by_class
        has_nrd_memory = class_name is not None and class_name in nrd_memory_by_class
        if (args.enable_nrs and not has_nrs_memory) or (args.enable_nrd and not has_nrd_memory):
            if not getattr(runtime_args, "_fewshot_nrs_warning_printed", False):
                print(f"[LEC][WARN] reference calibration skipped for class={class_name}")
                runtime_args._fewshot_nrs_warning_printed = True
            return outputs

        selected_map = np.asarray(outputs[pgcre.MS_METHOD]["map"], dtype=np.float32)
        original_image_score = float(outputs[pgcre.MS_METHOD].get("score", 0.0))
        final_map = selected_map
        if args.enable_nrs:
            dense_feat, real_h, real_w, _ = pgcre.encode_dense_image(
                image=image,
                image_processor=baseline_ctx.image_processor,
                model=baseline_ctx.model,
                resize_target=runtime_args.resize_short_edge,
                max_num_patches=runtime_args.max_num_patches,
                feature_layer=0,
            )
            memory = nrs_memory_by_class[class_name].to(dense_feat.device)
            normal_sim = []
            chunk_size = max(1024, int(getattr(runtime_args, "good_bank_chunk_size", 8192)))
            for start in range(0, dense_feat.shape[0], chunk_size):
                chunk = dense_feat[start : start + chunk_size].float()
                sim = chunk @ memory.T
                normal_sim.append(sim.max(dim=1).values.detach().float().cpu())
            sim_map = torch.cat(normal_sim, dim=0).numpy().reshape(real_h, real_w).astype(np.float32)
            sim_resized = pgcre.resize_map(sim_map, (selected_map.shape[1], selected_map.shape[0])).astype(np.float32)
            sim_norm = normalize_map(sim_resized)
            tau = min(max(float(args.nrs_tau), 0.0), 0.999)
            if tau > 0.0:
                sim_suppress = np.clip((sim_norm - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)
            else:
                sim_suppress = sim_norm
            power = max(float(args.nrs_power), 1e-6)
            if abs(power - 1.0) > 1e-6:
                sim_suppress = np.power(sim_suppress, power)
            if args.nrs_mode == "subtractive":
                final_map = final_map - float(args.nrs_alpha) * sim_suppress
            else:
                final_map = final_map * np.power(1.0 - sim_suppress, float(args.nrs_gamma))
            outputs["debug"]["nrs_similarity"] = sim_norm
        if args.enable_nrd:
            route_mode = class_route.get(class_name, "nrd") if class_name is not None else "nrd"
            if route_mode in {"pgcre", "off", "disable", "baseline"}:
                outputs["debug"]["nrd_route_mode"] = route_mode
                outputs["debug"]["nrd_enabled"] = False
                final_map = np.nan_to_num(final_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                score = topk_score_from_map(final_map, getattr(runtime_args, "pars_topk_ratio", 0.05))
                outputs[pgcre.MS_METHOD]["map"] = final_map
                outputs[pgcre.MS_METHOD]["score"] = score
                outputs["debug"]["selected_map"] = final_map
                outputs["debug"]["final_heatmap"] = final_map
                outputs["debug"]["nrs_enabled"] = bool(args.enable_nrs)
                return outputs
            if args.nrd_zscore != "image":
                print("[LEC][WARN] using image-level normalization in low-storage mode")
            nrd_feat, nrd_h, nrd_w, _ = pgcre.encode_dense_image(
                image=image,
                image_processor=baseline_ctx.image_processor,
                model=baseline_ctx.model,
                resize_target=runtime_args.resize_short_edge,
                max_num_patches=runtime_args.max_num_patches,
                feature_layer=int(args.nrd_feature_layer),
            )
            distance = compute_normal_reference_distance(
                nrd_feat,
                nrd_memory_by_class[class_name],
                chunk_size=max(1024, int(getattr(runtime_args, "good_bank_chunk_size", 8192))),
                eps=float(args.nrd_eps),
            ).numpy()
            d_map = resize_distance_to_heatmap(distance, (nrd_h, nrd_w), final_map.shape)
            normal_prior_map = 1.0 - normalize_map(d_map)
            if args.nrd_only:
                final_map, fusion_details = fuse_pgcre_with_nrd(
                    np.zeros_like(final_map, dtype=np.float32),
                    d_map,
                    beta=1.0,
                    fusion_mode="add",
                    eps=float(args.nrd_eps),
                    return_details=True,
                )
            else:
                final_map, fusion_details = fuse_pgcre_with_nrd(
                    final_map,
                    d_map,
                    beta=float(args.nrd_beta),
                    fusion_mode=args.nrd_fusion_mode,
                    eps=float(args.nrd_eps),
                    pro_q=float(args.nrd_pro_q),
                    pro_beta=float(args.nrd_pro_beta),
                    eta_nrd=float(args.nrd_eta_nrd),
                    lambda_prior=float(args.nrd_lambda_prior),
                    normal_prior_map=normal_prior_map,
                    agree_tau=float(args.nrd_agree_tau),
                    return_details=True,
                )
            outputs["debug"]["nrd_distance"] = d_map.astype(np.float32)
            outputs["debug"]["nrd_enabled"] = True
            outputs["debug"]["nrd_route_mode"] = route_mode
            outputs["debug"]["nrd_candidate_mask"] = fusion_details["candidate_mask"]
            outputs["debug"]["nrd_normal_prior"] = fusion_details["normal_prior"]
            outputs["debug"]["nrd_foreground_map"] = fusion_details["foreground_map"]
            outputs["debug"]["nrd_background_map"] = fusion_details["background_map"]
            outputs["debug"]["nrd_fps_norm"] = fusion_details["h_nrd_fps_norm"]
            outputs["debug"]["pgcre_norm"] = fusion_details["h_pgcre_norm"]
        final_map = np.nan_to_num(final_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        score = topk_score_from_map(final_map, getattr(runtime_args, "pars_topk_ratio", 0.05))
        if args.enable_nrd and args.nrd_fusion_mode in {"candidate_pgcre_rank", "rank_residual", "agreement_gated_add"}:
            # Keep image-level ranking anchored to the original PGCRE score.
            score = original_image_score
        outputs[pgcre.MS_METHOD]["map"] = final_map
        outputs[pgcre.MS_METHOD]["score"] = score
        outputs["debug"]["selected_map"] = final_map
        outputs["debug"]["final_heatmap"] = final_map
        outputs["debug"]["nrs_enabled"] = bool(args.enable_nrs)
        return outputs

    pgcre.build_good_prototype = build_good_prototype_from_sampled
    mvbench.build_good_prototype = build_good_prototype_from_sampled
    pgcre.infer_ms_fb_mg = infer_ms_fb_mg_with_nrs
    pgcre.Image.open = open_with_class_tag
    pgcre.load_model = load_model_with_token_modulator
    if args.fp_metrics_only:
        def compute_subset_fp_only(
            records,
            method,
            subset,
            pro_limit,
            metric_exact_max_pixels,
            metric_bins,
        ):
            if subset == "all":
                chosen = records
            else:
                anomalous = [
                    item
                    for item in records
                    if item["label"] == 1 and subset in item["subsets"]
                ]
                normal = [item for item in records if item["label"] == 0]
                chosen = [*normal, *anomalous]
            class_name = active_reference_class.get("value") or "unknown"
            values = _compute_fp_metrics(chosen, method, fp_thresholds)
            values.update(
                {
                    "class": class_name,
                    "method": method,
                    "subset": subset,
                }
            )
            fp_rows.append(values)
            return {
                "num_samples": len(chosen),
                "au_pro": float("nan"),
                "seg_auc": float("nan"),
                "seg_f1": float("nan"),
                "pixel_ap": float("nan"),
                "cls_auc": float("nan"),
                "cls_f1": float("nan"),
            }

        pgcre.compute_subset_metrics = compute_subset_fp_only

    argv_backup = sys.argv[:]
    sys.argv = [
        "sarc_runtime.py",
        "--dataset",
        args.dataset,
        "--data_root",
        str(args.data_root),
        "--model_path",
        str(args.model_path),
        "--mg_model_path",
        str(args.mg_model_path),
        "--output_dir",
        str(args.output_dir),
        "--method",
        "ms_fb_mg_fgclip",
        "--method_name",
        "Ours",
        "--subset",
        args.subset,
        "--tile_mode",
        args.tile_mode,
        "--overlap",
        str(args.overlap),
        "--lambda_weight",
        str(args.lambda_weight),
        "--q",
        str(args.q),
        "--prompt_ensemble",
        "--prompt_style",
        args.prompt_style,
        "--mg_fusion_mode",
        str(args.mg_fusion_mode),
        "--mg_fusion_weight",
        str(args.mg_fusion_weight),
        "--foreground_mode",
        "heatmap_topk",
        "--mg_start_layer",
        "3",
        "--mg_mask_threshold",
        "0.5",
        "--mg_mask_ratio",
        str(args.mg_mask_ratio),
        "--attention_bias_eta",
        "1.0",
        "--mg_neg_bias",
        "-10000",
        "--progress_style",
        "live",
    ]
    sys.argv.append(
        "--disable_mg" if args.disable_mg_branch else "--enable_mg"
    )
    sys.argv.append(
        "--disable_ms" if args.disable_ms_branch else "--enable_ms"
    )
    sys.argv.append("--enable_fg" if args.enable_fg else "--disable_fg")
    sys.argv.append(
        "--enable_positive_fusion"
        if args.enable_positive_fusion
        else "--disable_positive_fusion"
    )
    if args.aupro_segauc_only:
        sys.argv.append("--aupro_segauc_only")
    if args.adaptive_prompt_policy_path is not None:
        sys.argv.extend(["--adaptive_prompt_policy_path", str(args.adaptive_prompt_policy_path)])
    if requested_classes:
        sys.argv.extend(["--classes", *requested_classes])
    if args.enable_dual_layer_guidance:
        sys.argv.extend(
            [
                "--dual_layer_guidance",
                "--local_feature_layer",
                str(args.local_feature_layer),
            ]
        )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pgcre.main()
    finally:
        sys.argv = argv_backup
        pgcre.build_good_prototype = original_pgcre_build
        mvbench.build_good_prototype = original_mv_build
        pgcre.infer_ms_fb_mg = original_infer_ms_fb_mg
        pgcre.Image.open = original_image_open
        pgcre.load_model = original_pgcre_load_model
        pgcre.compute_subset_metrics = original_compute_subset_metrics

    if args.fp_metrics_only:
        per_class_path = args.output_dir / "fp_metrics_per_class.csv"
        summary_path = args.output_dir / "fp_metrics_summary.json"
        _write_csv(per_class_path, fp_rows)
        summary = _summarize_fp_rows(fp_rows, fp_thresholds)
        summary.update(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "shots": args.shots,
                "token_modulator_checkpoint": (
                    str(args.token_modulator_checkpoint)
                    if args.token_modulator_checkpoint is not None
                    else None
                ),
            }
        )
        _save_json(summary_path, summary)
        print(f"[FP] per-class metrics: {per_class_path}")
        print(f"[FP] summary: {summary_path}")

    metrics_path = args.output_dir / "metrics.json"
    metrics = _load_json(metrics_path)
    metrics.update(
        {
            "method": "Ours",
            "fewshot_role": "sampled_normal_calibration",
            "sampled_normals": str(args.sampled_normals),
            "enable_fg": bool(args.enable_fg),
            "enable_positive_fusion": bool(args.enable_positive_fusion),
            "mg_fusion_mode": args.mg_fusion_mode,
            "mg_fusion_weight": float(args.mg_fusion_weight),
            "enable_nrs": bool(args.enable_nrs),
            "nrs_mode": args.nrs_mode,
            "nrs_alpha": float(args.nrs_alpha),
            "nrs_gamma": float(args.nrs_gamma),
            "nrs_tau": float(args.nrs_tau),
            "nrs_power": float(args.nrs_power),
            "enable_nrd": bool(args.enable_nrd),
            "nrd_beta": float(args.nrd_beta),
            "nrd_fusion_mode": args.nrd_fusion_mode,
            "adaptive_prompt_policy_path": str(args.adaptive_prompt_policy_path) if args.adaptive_prompt_policy_path is not None else None,
            "aupro_segauc_only": bool(args.aupro_segauc_only),
            "nrd_feature_layer": int(args.nrd_feature_layer),
            "nrd_max_ref_patches": int(args.nrd_max_ref_patches),
            "nrd_zscore": args.nrd_zscore,
            "nrd_eps": float(args.nrd_eps),
            "nrd_pro_q": float(args.nrd_pro_q),
            "nrd_pro_beta": float(args.nrd_pro_beta),
            "nrd_eta_nrd": float(args.nrd_eta_nrd),
            "nrd_lambda_prior": float(args.nrd_lambda_prior),
            "nrd_agree_tau": float(args.nrd_agree_tau),
            "nrd_class_route_json": str(args.nrd_class_route_json) if args.nrd_class_route_json is not None else None,
            "nrd_use_fgclip_patch_features": bool(args.nrd_use_fgclip_patch_features),
            "nrd_only": bool(args.nrd_only),
            "status": "ok",
        }
    )
    _save_json(args.output_dir / "eval_metrics.json", metrics)
    print(f"[SARC] metrics: {args.output_dir / 'eval_metrics.json'}")
    if args.enable_nrd:
        print("[LEC] normal-reference calibration ready")


if __name__ == "__main__":
    main()
