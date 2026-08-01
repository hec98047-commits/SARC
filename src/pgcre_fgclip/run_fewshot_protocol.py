from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fewshot_sampler import sample_fewshot_normals, save_sampled_normals


METHOD_LABELS = {
    "fgclip": "FG-CLIP",
    "winclip": "WinCLIP",
    "patchcore": "PatchCore",
    "ours": "Ours",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --config files. Install pyyaml or pass CLI args directly.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_csv_list(value, cast=str):
    if value is None:
        return None
    if isinstance(value, list):
        return [cast(item) for item in value]
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _str_bool(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Run low-storage few-shot normal calibration protocol.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--mg_model_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--shots_list", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument(
        "--modules",
        default=None,
        help="Comma-separated method modules to enable: A,B,C.",
    )
    parser.add_argument("--subset", choices=["all", "small", "tiny"], default=None)
    parser.add_argument("--low_storage", default=None)
    parser.add_argument("--classes", default=None, help="Optional comma-separated class list for quick debugging.")
    parser.add_argument("--sampling_policy", choices=["random", "promptad"], default=None)
    parser.add_argument("--promptad_root", type=Path, default=None)
    parser.add_argument("--enable_dual_layer_guidance", action="store_true")
    parser.add_argument("--local_feature_layer", type=int, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--lambda_weight", type=float, default=None)
    parser.add_argument("--tile_mode", default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--prompt_style", choices=["default", "spatial_aware", "adaptive", "pcb_specific"], default=None)
    parser.add_argument("--adaptive_prompt_policy_path", type=Path, default=None)
    parser.add_argument("--token_modulator_checkpoint", type=Path, default=None)
    parser.add_argument("--enable_nrs", action="store_true")
    parser.add_argument("--nrs_mode", choices=["subtractive", "multiplicative"], default=None)
    parser.add_argument("--nrs_alpha", type=float, default=None)
    parser.add_argument("--nrs_gamma", type=float, default=None)
    parser.add_argument("--nrs_tau", type=float, default=None)
    parser.add_argument("--nrs_power", type=float, default=None)
    parser.add_argument("--enable_nrd", action="store_true")
    parser.add_argument("--nrd_beta", type=float, default=None)
    parser.add_argument("--nrd_fusion_mode", choices=["add", "residual", "multiply"], default=None)
    parser.add_argument("--nrd_feature_layer", type=int, default=None)
    parser.add_argument("--nrd_max_ref_patches", type=int, default=None)
    parser.add_argument("--nrd_zscore", choices=["image", "class"], default=None)
    parser.add_argument("--nrd_eps", type=float, default=None)
    parser.add_argument("--nrd_class_route_json", type=Path, default=None)
    parser.add_argument("--nrd_use_fgclip_patch_features", action="store_true")
    parser.add_argument("--nrd_only", action="store_true")
    return parser.parse_args()


def resolve_config(args) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg.update(_load_yaml(args.config))
    for key in [
        "dataset",
        "data_root",
        "model_path",
        "mg_model_path",
        "output_dir",
        "subset",
        "low_storage",
        "classes",
        "sampling_policy",
        "promptad_root",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.shots_list is not None:
        cfg["shots_list"] = _parse_csv_list(args.shots_list, int)
    if args.seeds is not None:
        cfg["seeds"] = _parse_csv_list(args.seeds, int)
    if args.methods is not None:
        cfg["methods"] = _parse_csv_list(args.methods, str)
    if args.modules is not None:
        cfg["modules"] = [item.upper() for item in _parse_csv_list(args.modules, str)]
    if args.enable_dual_layer_guidance:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["enable_dual_layer_guidance"] = True
    if args.local_feature_layer is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["local_feature_layer"] = int(args.local_feature_layer)
    if args.q is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["q"] = float(args.q)
    if args.lambda_weight is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["lambda_weight"] = float(args.lambda_weight)
    if args.tile_mode is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["tile_mode"] = str(args.tile_mode)
    if args.overlap is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["overlap"] = float(args.overlap)
    if args.prompt_style is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["prompt_style"] = str(args.prompt_style)
    if args.adaptive_prompt_policy_path is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["adaptive_prompt_policy_path"] = Path(args.adaptive_prompt_policy_path)
    if args.token_modulator_checkpoint is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["token_modulator_checkpoint"] = Path(args.token_modulator_checkpoint)
    if args.enable_nrs:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["enable_nrs"] = True
    if args.enable_nrd:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["enable_nrd"] = True
    if args.nrd_only:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["nrd_only"] = True
    if args.nrd_use_fgclip_patch_features:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["nrd_use_fgclip_patch_features"] = True
    for key in ["nrs_mode", "nrs_alpha", "nrs_gamma", "nrs_tau", "nrs_power"]:
        value = getattr(args, key)
        if value is not None:
            cfg.setdefault("pgcre", {})
            cfg["pgcre"][key] = value
    if args.nrd_beta is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["nrd_beta"] = float(args.nrd_beta)
    for key in ["nrd_fusion_mode", "nrd_feature_layer", "nrd_max_ref_patches", "nrd_zscore", "nrd_eps"]:
        value = getattr(args, key)
        if value is not None:
            cfg.setdefault("pgcre", {})
            cfg["pgcre"][key] = value
    if args.nrd_class_route_json is not None:
        cfg.setdefault("pgcre", {})
        cfg["pgcre"]["nrd_class_route_json"] = Path(args.nrd_class_route_json)

    required = ["dataset", "data_root", "model_path", "mg_model_path", "output_dir"]
    missing = [key for key in required if key not in cfg or cfg[key] in {None, ""}]
    if missing:
        raise ValueError(f"Missing required few-shot config fields: {missing}")

    cfg["data_root"] = Path(cfg["data_root"])
    cfg["model_path"] = Path(cfg["model_path"])
    cfg["mg_model_path"] = Path(cfg["mg_model_path"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg["shots_list"] = _parse_csv_list(cfg.get("shots_list", [1, 2, 4, 8]), int)
    cfg["seeds"] = _parse_csv_list(cfg.get("seeds", [42]), int)
    cfg["methods"] = [item.lower() for item in _parse_csv_list(cfg.get("methods", ["fgclip", "winclip", "patchcore", "ours"]), str)]
    cfg["modules"] = {
        str(item).strip().upper()
        for item in _parse_csv_list(cfg.get("modules", ["A", "B", "C"]), str)
    }
    invalid_modules = cfg["modules"] - {"A", "B", "C"}
    if invalid_modules:
        raise ValueError(f"Unknown modules: {sorted(invalid_modules)}")
    cfg["subset"] = cfg.get("subset", "all")
    cfg["low_storage"] = _str_bool(cfg.get("low_storage", True))
    cfg["sampling_policy"] = str(cfg.get("sampling_policy", "random")).lower()
    if cfg.get("promptad_root") is not None:
        cfg["promptad_root"] = Path(cfg["promptad_root"])
    cfg["pgcre"] = cfg.get("pgcre", {}) or {}
    cfg["pgcre"].setdefault(
        "token_modulator_checkpoint",
        "checkpoints/{dataset}_{shot}shot_seed{seed}/ctm_epoch10_trained.pt",
    )
    return cfg


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric(payload: dict, key: str):
    aliases = {
        "mean_seg_auc": ["mean_seg_auc", "mean_segmentation_au_roc", "seg_auc"],
        "mean_au_pro": ["mean_au_pro", "au_pro"],
        "mean_seg_f1": ["mean_seg_f1", "mean_segmentation_f1_max", "seg_f1"],
        "mean_pixel_ap": ["mean_pixel_ap", "pixel_ap"],
        "mean_cls_auc": ["mean_cls_auc", "mean_classification_au_roc", "cls_auc"],
        "mean_cls_f1": ["mean_cls_f1", "mean_classification_f1_max", "cls_f1"],
    }
    for alias in aliases[key]:
        if alias in payload:
            return payload[alias]
    return float("nan")


def run_method(method: str, cfg: dict[str, Any], shot: int, seed: int, sampled_path: Path, method_dir: Path) -> dict:
    script_map = {
        "fgclip": "eval_fewshot_fgclip.py",
        "winclip": "eval_fewshot_winclip.py",
        "patchcore": "eval_fewshot_patchcore.py",
        "ours": "eval_fewshot_ours_pgcre.py",
    }
    script = script_map[method]
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / script),
        "--dataset",
        cfg["dataset"],
        "--data_root",
        str(cfg["data_root"]),
        "--sampled_normals",
        str(sampled_path),
        "--output_dir",
        str(method_dir),
        "--subset",
        cfg["subset"],
        "--low_storage",
        cfg["low_storage"],
    ]
    if method in {"fgclip", "patchcore", "ours"}:
        cmd.extend(["--model_path", str(cfg["model_path"])])
    if method == "ours":
        pgcre = cfg.get("pgcre", {})
        enabled_modules = cfg.get("modules", {"A", "B", "C"})
        module_a_enabled = "A" in enabled_modules
        module_b_enabled = "B" in enabled_modules
        module_c_enabled = "C" in enabled_modules
        if module_c_enabled and not module_b_enabled:
            raise ValueError("Module C requires the candidate-response path provided by Module B.")

        cmd.extend(["--mg_model_path", str(cfg["mg_model_path"])])
        if cfg.get("classes"):
            cmd.extend(["--classes", str(cfg["classes"])])
        for key in ["tile_mode", "overlap", "lambda_weight", "q"]:
            if key in pgcre:
                cmd.extend([f"--{key}", str(pgcre[key])])
        prompt_style = str(pgcre.get("prompt_style", "spatial_aware")) if module_a_enabled else "default"
        cmd.extend(["--prompt_style", prompt_style])
        if module_a_enabled and "adaptive_prompt_policy_path" in pgcre:
            cmd.extend(["--adaptive_prompt_policy_path", str(pgcre["adaptive_prompt_policy_path"])])
        if not module_b_enabled:
            cmd.append("--disable_mg_branch")
        if not module_c_enabled:
            cmd.append("--disable_ms_branch")
            cmd.extend(["--mg_fusion_mode", "direct"])
        else:
            cmd.extend(["--mg_fusion_mode", "positive"])
            cmd.extend(["--enable_fg", "--enable_positive_fusion"])
        if module_b_enabled:
            checkpoint_template = pgcre.get("token_modulator_checkpoint")
            if checkpoint_template is None:
                raise ValueError(
                    "Module B requires pgcre.token_modulator_checkpoint. "
                    "Train CandidateTokenModulator first."
                )
            checkpoint_path = Path(
                str(checkpoint_template).format(
                    dataset=cfg["dataset"],
                    shot=shot,
                    seed=seed,
                )
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Module B checkpoint not found: {checkpoint_path}. "
                    "Run src/pgcre_fgclip/train_candidate_token_modulator.py first."
                )
            cmd.extend(["--token_modulator_checkpoint", str(checkpoint_path)])
        if bool(pgcre.get("enable_dual_layer_guidance", False)):
            cmd.append("--enable_dual_layer_guidance")
            cmd.extend(["--local_feature_layer", str(pgcre.get("local_feature_layer", 5))])
        if bool(pgcre.get("enable_nrs", False)):
            cmd.append("--enable_nrs")
            cmd.extend(["--nrs_mode", str(pgcre.get("nrs_mode", "subtractive"))])
            cmd.extend(["--nrs_alpha", str(pgcre.get("nrs_alpha", 0.2))])
            cmd.extend(["--nrs_gamma", str(pgcre.get("nrs_gamma", 1.0))])
            cmd.extend(["--nrs_tau", str(pgcre.get("nrs_tau", 0.0))])
            cmd.extend(["--nrs_power", str(pgcre.get("nrs_power", 1.0))])
        if module_b_enabled and bool(pgcre.get("enable_nrd", True)):
            cmd.append("--enable_nrd")
            cmd.extend(["--nrd_beta", str(pgcre.get("nrd_beta", 0.3))])
            cmd.extend(["--nrd_fusion_mode", str(pgcre.get("nrd_fusion_mode", "add"))])
            cmd.extend(["--nrd_feature_layer", str(pgcre.get("nrd_feature_layer", 5))])
            cmd.extend(["--nrd_max_ref_patches", str(pgcre.get("nrd_max_ref_patches", 4096))])
            cmd.extend(["--nrd_ref_selection", str(pgcre.get("nrd_ref_selection", "fps"))])
            cmd.extend(["--nrd_zscore", str(pgcre.get("nrd_zscore", "image"))])
            cmd.extend(["--nrd_eps", str(pgcre.get("nrd_eps", 1e-6))])
            if "nrd_class_route_json" in pgcre:
                cmd.extend(["--nrd_class_route_json", str(pgcre["nrd_class_route_json"])])
            if bool(pgcre.get("nrd_use_fgclip_patch_features", False)):
                cmd.append("--nrd_use_fgclip_patch_features")
        if bool(pgcre.get("nrd_only", False)):
            cmd.append("--nrd_only")
    subprocess.run(cmd, check=True)

    metrics_path = method_dir / "eval_metrics.json"
    payload = _load_json(metrics_path)
    return {
        "dataset": cfg["dataset"],
        "shots": int(shot),
        "seed": int(seed),
        "method": METHOD_LABELS[method],
        "status": payload.get("status", "ok"),
        "mean_seg_auc": _metric(payload, "mean_seg_auc"),
        "mean_au_pro": _metric(payload, "mean_au_pro"),
        "mean_seg_f1": _metric(payload, "mean_seg_f1"),
        "mean_pixel_ap": _metric(payload, "mean_pixel_ap"),
        "mean_cls_auc": _metric(payload, "mean_cls_auc"),
        "mean_cls_f1": _metric(payload, "mean_cls_f1"),
    }


def write_wide_tables(output_dir: Path, rows: list[dict]):
    metric_specs = [
        ("mean_pixel_ap", "fewshot_table_pixelap.csv"),
        ("mean_seg_f1", "fewshot_table_segf1.csv"),
        ("mean_seg_auc", "fewshot_table_segauc.csv"),
        ("mean_au_pro", "fewshot_table_aupro.csv"),
    ]
    methods = ["FG-CLIP", "WinCLIP", "PatchCore", "Ours"]
    shots = sorted({int(row["shots"]) for row in rows})
    for metric_key, file_name in metric_specs:
        table_rows = []
        for shot in shots:
            row = {"Normal samples per class": f"{shot}-shot"}
            for method in methods:
                candidates = [item for item in rows if int(item["shots"]) == shot and item["method"] == method]
                row[method] = candidates[0][metric_key] if candidates else "nan"
            table_rows.append(row)
        _write_csv(output_dir / file_name, table_rows, ["Normal samples per class", *methods])


def main():
    args = parse_args()
    cfg = resolve_config(args)
    root = cfg["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for seed in cfg["seeds"]:
        for shot in cfg["shots_list"]:
            sample_dir = root / "samples" / f"{shot}shot"
            if len(cfg["seeds"]) > 1:
                sample_dir = root / "samples" / f"seed{seed}" / f"{shot}shot"
            sampled_path = sample_dir / "sampled_normal_paths.json"
            if not sampled_path.exists():
                payload = sample_fewshot_normals(
                    cfg["dataset"],
                    cfg["data_root"],
                    int(shot),
                    int(seed),
                    sampling_policy=cfg.get("sampling_policy", "random"),
                    promptad_root=cfg.get("promptad_root"),
                )
                save_sampled_normals(payload, sampled_path)
            print(f"[FewShot] sampled normals ready: {sampled_path}")

            for method in cfg["methods"]:
                if method not in METHOD_LABELS:
                    print(f"[FewShot][WARN] Unknown method skipped: {method}")
                    continue
                method_dir = root / method / f"{shot}shot"
                if len(cfg["seeds"]) > 1:
                    method_dir = root / method / f"{shot}shot" / f"seed{seed}"
                row = run_method(method, cfg, int(shot), int(seed), sampled_path, method_dir)
                all_rows.append(row)

    fields = [
        "dataset",
        "shots",
        "seed",
        "method",
        "status",
        "mean_seg_auc",
        "mean_au_pro",
        "mean_seg_f1",
        "mean_pixel_ap",
        "mean_cls_auc",
        "mean_cls_f1",
    ]
    _write_csv(root / "fewshot_summary.csv", all_rows, fields)
    write_wide_tables(root, all_rows)
    print("[FewShot] sampler ready.")
    print("[FewShot] FG-CLIP baseline ready.")
    print("[FewShot] Ours PGCRE few-shot ready.")
    print("[FewShot] PatchCore few-shot ready or marked missing.")
    print("[FewShot] WinCLIP few-shot ready or marked missing.")
    print("[FewShot] summary tables saved.")


if __name__ == "__main__":
    main()
