from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fgclip_ad.datasets import discover_classes, load_dataset_samples


def _parse_promptad_seed_indices(seed_file: Path, shots: int) -> list[int]:
    begin_str = f"#{int(shots)}: "
    with open(seed_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith(begin_str):
                continue
            suffix = line[len(begin_str) :].strip()
            if not suffix:
                return []
            return [int(item) for item in suffix.split() if item.strip()]
    raise RuntimeError(f"PromptAD seed file does not contain a {shots}-shot entry: {seed_file}")


def _sample_with_promptad_seeds(
    dataset: str,
    data_root: Path,
    shots: int,
    promptad_root: Path,
) -> dict[str, list[str]]:
    dataset = str(dataset).lower()
    seed_dir_name = "seeds_mvtec" if dataset == "mvtec" else "seeds_visa"
    seed_root = Path(promptad_root) / "datasets" / seed_dir_name
    if not seed_root.exists():
        raise FileNotFoundError(f"PromptAD seed directory not found: {seed_root}")

    classes: dict[str, list[str]] = {}
    prepared_visa_root = promptad_root / "DATA" / "anomaly_detection" / "VisA_pytorch" / "1cls"
    for class_name in discover_classes(dataset, data_root):
        train_samples, _ = load_dataset_samples(dataset, data_root, class_name)
        normal_paths = sorted([str(item.image_path) for item in train_samples if not item.is_anomaly])
        seed_file = seed_root / class_name / "selected_samples_per_run.txt"
        if not seed_file.exists():
            raise FileNotFoundError(f"PromptAD seed file not found for class '{class_name}': {seed_file}")
        indices = _parse_promptad_seed_indices(seed_file, shots)
        selected = []
        if dataset == "visa" and prepared_visa_root.exists():
            promptad_class_good_dir = prepared_visa_root / class_name / "train" / "good"
            promptad_normal_paths = sorted([str(path) for path in promptad_class_good_dir.glob("*.JPG")])
            raw_path_by_name = {Path(path).name.lower(): path for path in normal_paths}
            for index in indices:
                if index < 0 or index >= len(promptad_normal_paths):
                    raise RuntimeError(
                        f"PromptAD index out of range for visa/{class_name}: "
                        f"index={index}, num_promptad_normals={len(promptad_normal_paths)}, seed_file={seed_file}"
                    )
                promptad_name = Path(promptad_normal_paths[index]).name.lower()
                mapped = raw_path_by_name.get(promptad_name)
                if mapped is None:
                    raise RuntimeError(
                        f"Could not map PromptAD-selected VisA sample back to raw data for {class_name}: "
                        f"filename={promptad_name}"
                    )
                selected.append(mapped)
        else:
            for index in indices:
                if index < 0 or index >= len(normal_paths):
                    raise RuntimeError(
                        f"PromptAD index out of range for {dataset}/{class_name}: "
                        f"index={index}, num_normals={len(normal_paths)}, seed_file={seed_file}"
                    )
                selected.append(normal_paths[index])
        classes[class_name] = sorted(selected)
    return classes


def sample_fewshot_normals(
    dataset: str,
    data_root: str | Path,
    shots: int,
    seed: int = 42,
    sampling_policy: str = "random",
    promptad_root: str | Path | None = None,
) -> dict:
    dataset = str(dataset).lower()
    data_root = Path(data_root)
    sampling_policy = str(sampling_policy).lower()
    if sampling_policy == "promptad":
        if promptad_root is None:
            raise ValueError("promptad_root is required when sampling_policy='promptad'.")
        classes = _sample_with_promptad_seeds(dataset, data_root, shots, Path(promptad_root))
    else:
        rng = random.Random(int(seed))
        classes = {}
        for class_name in discover_classes(dataset, data_root):
            train_samples, _ = load_dataset_samples(dataset, data_root, class_name)
            normal_paths = sorted([str(item.image_path) for item in train_samples if not item.is_anomaly])
            if len(normal_paths) < int(shots):
                print(
                    f"[FewShot][WARN] {dataset}/{class_name} has only {len(normal_paths)} "
                    f"normal train images; requested {shots}."
                )
                selected = normal_paths
            else:
                selected = sorted(rng.sample(normal_paths, int(shots)))
            classes[class_name] = selected

    return {
        "dataset": dataset,
        "data_root": str(data_root),
        "shots": int(shots),
        "seed": int(seed),
        "sampling_policy": sampling_policy,
        "promptad_root": str(Path(promptad_root)) if promptad_root is not None else None,
        "classes": classes,
    }


def save_sampled_normals(payload: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample K normal train/good images per class.")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_policy", choices=["random", "promptad"], default="random")
    parser.add_argument("--promptad_root", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = sample_fewshot_normals(
        args.dataset,
        args.data_root,
        args.shots,
        args.seed,
        sampling_policy=args.sampling_policy,
        promptad_root=args.promptad_root,
    )
    path = save_sampled_normals(payload, args.output_path)
    print(f"[FewShot] sampled_normal_paths = {path}")


if __name__ == "__main__":
    main()
