from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PAPER_STAGES = ("SP", "ARC", "LEC")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for SARC release configs.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _csv_values(value: Any, cast=str) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [cast(item) for item in value]
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SARC few-shot protocol with SP, ARC, and LEC."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=["mvtec", "visa"])
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--sarc_model_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--shots_list")
    parser.add_argument("--seeds")
    parser.add_argument("--stages", help="Comma-separated SARC stages: SP,ARC,LEC.")
    parser.add_argument("--subset", choices=["all", "small", "tiny"])
    parser.add_argument("--classes")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_yaml(args.config) if args.config else {}
    for key in (
        "dataset",
        "data_root",
        "model_path",
        "sarc_model_path",
        "output_dir",
        "subset",
        "classes",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.shots_list is not None:
        cfg["shots_list"] = _csv_values(args.shots_list, int)
    if args.seeds is not None:
        cfg["seeds"] = _csv_values(args.seeds, int)
    if args.stages is not None:
        cfg["stages"] = _csv_values(args.stages, str)

    required = ("dataset", "data_root", "model_path", "sarc_model_path", "output_dir")
    missing = [key for key in required if cfg.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required SARC config fields: {missing}")

    for key in ("data_root", "model_path", "sarc_model_path", "output_dir"):
        cfg[key] = Path(cfg[key])
    cfg["shots_list"] = _csv_values(cfg.get("shots_list", [1, 2, 4]), int)
    cfg["seeds"] = _csv_values(cfg.get("seeds", [42]), int)
    cfg["stages"] = {item.strip().upper() for item in _csv_values(cfg.get("stages", PAPER_STAGES))}
    unknown = cfg["stages"] - set(PAPER_STAGES)
    if unknown:
        raise ValueError(f"Unknown SARC stages: {sorted(unknown)}")
    if "LEC" in cfg["stages"] and "ARC" not in cfg["stages"]:
        raise ValueError("LEC requires the constrained response produced by ARC.")
    cfg["subset"] = str(cfg.get("subset", "all"))
    cfg["sarc"] = cfg.get("sarc", {}) or {}
    return cfg


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    aliases = {
        "p_auroc": ("mean_seg_auc", "mean_segmentation_au_roc", "seg_auc"),
        "au_pro": ("mean_au_pro", "au_pro"),
    }
    row: dict[str, Any] = {"status": payload.get("status", "ok")}
    for output_key, candidates in aliases.items():
        row[output_key] = next((payload[key] for key in candidates if key in payload), float("nan"))
    return row


def _run_one(cfg: dict[str, Any], shot: int, seed: int, samples: Path, output_dir: Path) -> dict[str, Any]:
    options = cfg["sarc"]
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_fewshot_sarc.py"),
        "--dataset", str(cfg["dataset"]),
        "--data_root", str(cfg["data_root"]),
        "--model_path", str(cfg["model_path"]),
        "--sarc_model_path", str(cfg["sarc_model_path"]),
        "--sampled_normals", str(samples),
        "--output_dir", str(output_dir),
        "--subset", cfg["subset"],
        "--shots", str(shot),
        "--seed", str(seed),
        "--tile_mode", str(options.get("tile_mode", "2x2")),
        "--overlap", str(options.get("overlap", 0.25)),
        "--lambda_weight", str(options.get("lambda_weight", 0.5)),
        "--q", str(options.get("q", 0.8)),
    ]
    if cfg.get("classes"):
        command.extend(["--classes", str(cfg["classes"])])

    stages = cfg["stages"]
    if "SP" not in stages:
        command.extend(["--prompt_style", "default"])
    else:
        command.extend(["--prompt_style", str(options.get("prompt_style", "spatial_aware"))])
        if options.get("adaptive_prompt_policy_path"):
            command.extend(["--adaptive_prompt_policy_path", str(options["adaptive_prompt_policy_path"])])

    if "ARC" not in stages:
        command.append("--disable_arc")
    else:
        template = options.get("arc_checkpoint")
        if template:
            checkpoint = Path(str(template).format(dataset=cfg["dataset"], shot=shot, seed=seed))
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"ARC checkpoint not found: {checkpoint}. Run src/sarc/train_arc.py first."
                )
            command.extend(["--arc_checkpoint", str(checkpoint)])

    if "LEC" not in stages:
        command.append("--disable_lec")
    elif bool(options.get("normal_reference_calibration", True)):
        command.extend([
            "--normal_reference_calibration",
            "--reference_selection", str(options.get("reference_selection", "fps")),
        ])

    subprocess.run(command, check=True)
    metrics = _load_metrics(output_dir / "eval_metrics.json")
    return {
        "dataset": cfg["dataset"],
        "shots": shot,
        "seed": seed,
        "method": "SARC",
        **metrics,
    }


def main() -> None:
    cfg = resolve_config(parse_args())
    from fewshot_sampler import sample_fewshot_normals, save_sampled_normals

    root = cfg["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for seed in cfg["seeds"]:
        for shot in cfg["shots_list"]:
            sample_path = root / "samples" / f"seed{seed}" / f"{shot}shot" / "sampled_normal_paths.json"
            if not sample_path.is_file():
                sample_payload = sample_fewshot_normals(
                    cfg["dataset"], cfg["data_root"], shot, seed, sampling_policy="random"
                )
                save_sampled_normals(sample_payload, sample_path)
            run_dir = root / f"{shot}shot" / f"seed{seed}"
            rows.append(_run_one(cfg, shot, seed, sample_path, run_dir))

    summary = root / "sarc_fewshot_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "shots", "seed", "method", "status", "p_auroc", "au_pro"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SARC] results: {summary}")


if __name__ == "__main__":
    main()
