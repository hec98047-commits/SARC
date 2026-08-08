from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ARC token modulator using normal-only reference tokens."
    )
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default=None)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--sampled_normals", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--token_batch_size", type=int, default=2048)
    parser.add_argument("--max_num_patches", type=int, default=1024)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--separation_margin", type=float, default=0.02)
    parser.add_argument("--modulation_lambda", type=float, default=0.1)
    parser.add_argument("--preserve_weight", type=float, default=5.0)
    parser.add_argument("--bound_weight", type=float, default=0.25)
    parser.add_argument("--residual_ratio_bound", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_layer3_inputs(
    model,
    processor,
    image_paths: list[Path],
    max_num_patches: int,
) -> torch.Tensor:
    vision = model.vision_model
    collected: list[torch.Tensor] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            images=image,
            max_num_patches=max_num_patches,
            return_tensors="pt",
        ).to(model.device)
        hidden = vision.embeddings(inputs["pixel_values"], inputs["spatial_shapes"])
        pixel_mask = inputs.get("pixel_attention_mask")
        attention_mask = (
            _prepare_4d_attention_mask(pixel_mask, hidden.dtype)
            if pixel_mask is not None
            else None
        )
        for layer in vision.encoder.layers[:2]:
            hidden = layer(hidden, attention_mask)
        real_h, real_w = inputs["spatial_shapes"][0].tolist()
        collected.append(
            hidden[0, : int(real_h) * int(real_w)].detach().float().cpu()
        )
    return torch.cat(collected, dim=0)


def adapter_state(adapter) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in adapter.state_dict().items()
    }


def save_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    epoch: int,
    num_reference_images: int,
    num_tokens: int,
    history: list[dict],
) -> None:
    torch.save(
        {
            "adapter": state,
            "seed": args.seed,
            "epochs": epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "objective": "candidate_separation_with_normal_preservation",
            "modulation_lambda": args.modulation_lambda,
            "noise_std": args.noise_std,
            "separation_margin": args.separation_margin,
            "preserve_weight": args.preserve_weight,
            "bound_weight": args.bound_weight,
            "residual_ratio_bound": args.residual_ratio_bound,
            "num_reference_images": num_reference_images,
            "num_tokens": num_tokens,
            "history": history,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print("[ARC] normal-only training", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    with args.sampled_normals.open("r", encoding="utf-8") as handle:
        sampled = json.load(handle)
    image_paths = [
        Path(path)
        for paths in sampled["classes"].values()
        for path in paths
    ]
    if not image_paths:
        raise RuntimeError("No sampled normal images were found.")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    ).cuda().eval()
    processor = AutoImageProcessor.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = model.vision_model.encoder.token_modulator
    if adapter is None:
        raise RuntimeError("The ARC token modulator is not present in the SARC model.")
    adapter.float().train()
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)

    tokens = collect_layer3_inputs(
        model,
        processor,
        image_paths,
        args.max_num_patches,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initial_state = adapter_state(adapter)
    save_checkpoint(
        args.output_dir / "arc_epoch00_initial.pt",
        initial_state,
        args,
        epoch=0,
        num_reference_images=len(image_paths),
        num_tokens=int(tokens.shape[0]),
        history=[],
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    permutation_generator = torch.Generator().manual_seed(args.seed)
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(
            tokens.shape[0],
            generator=permutation_generator,
        )
        records: list[tuple[float, float, float, float]] = []
        for start in range(0, tokens.shape[0], args.token_batch_size):
            batch = tokens[
                permutation[start : start + args.token_batch_size]
            ].cuda(non_blocking=True)
            prototype = F.normalize(batch.mean(dim=0), dim=0)
            perturbed = batch + args.noise_std * torch.randn_like(batch)

            before_distance = 1.0 - F.cosine_similarity(
                perturbed,
                prototype.unsqueeze(0),
                dim=-1,
            )
            mask_min = before_distance.min()
            mask_max = before_distance.max()
            soft_mask = (before_distance - mask_min) / (
                mask_max - mask_min
            ).clamp_min(1e-8)

            delta = adapter(perturbed)
            modulated = perturbed + (
                args.modulation_lambda * soft_mask.unsqueeze(-1) * delta
            )
            after_distance = 1.0 - F.cosine_similarity(
                modulated,
                prototype.unsqueeze(0),
                dim=-1,
            )

            loss_sep = F.relu(
                args.separation_margin
                - (after_distance - before_distance)
            ).mean()
            loss_preserve = (
                (
                    (1.0 - soft_mask).unsqueeze(-1)
                    * (modulated - perturbed)
                )
                .pow(2)
                .mean()
            )
            residual_ratio = delta.norm(dim=-1) / perturbed.norm(
                dim=-1
            ).clamp_min(1e-8)
            loss_bound = F.relu(
                residual_ratio - args.residual_ratio_bound
            ).pow(2).mean()
            loss = (
                loss_sep
                + args.preserve_weight * loss_preserve
                + args.bound_weight * loss_bound
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            records.append(
                (
                    float(loss.detach().cpu()),
                    float(loss_sep.detach().cpu()),
                    float(loss_preserve.detach().cpu()),
                    float(loss_bound.detach().cpu()),
                )
            )

        mean_values = np.asarray(records, dtype=np.float64).mean(axis=0)
        row = {
            "epoch": epoch,
            "loss": float(mean_values[0]),
            "separation": float(mean_values[1]),
            "preserve": float(mean_values[2]),
            "bound": float(mean_values[3]),
        }
        history.append(row)
        print(
            f"[ARC][epoch {epoch:02d}] "
            f"loss={row['loss']:.8f} "
            f"sep={row['separation']:.8f} "
            f"preserve={row['preserve']:.8e} "
            f"bound={row['bound']:.8f}",
            flush=True,
        )

    save_checkpoint(
        args.output_dir / f"arc_epoch{args.epochs:02d}.pt",
        adapter_state(adapter),
        args,
        epoch=args.epochs,
        num_reference_images=len(image_paths),
        num_tokens=int(tokens.shape[0]),
        history=history,
    )
    with (args.output_dir / "training_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "protocol": "ARC normal-only token-modulator training",
                "dataset": args.dataset,
                "seed": args.seed,
                "epochs": args.epochs,
                "num_reference_images": len(image_paths),
                "num_tokens": int(tokens.shape[0]),
                "history": history,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[ARC] training complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
