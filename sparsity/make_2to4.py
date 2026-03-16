"""Create exact 2:4 structured sparse masks/weights for linear layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def make_2to4_mask(weight: torch.Tensor) -> torch.Tensor:
    """Return a 0/1 mask with exact 2 non-zeros per contiguous group of 4 per row."""
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.shape}")

    rows, cols = weight.shape
    if cols % 4 != 0:
        raise ValueError(f"Column count must be divisible by 4, got {cols}")

    grouped = weight.abs().view(rows, cols // 4, 4)
    top2 = torch.topk(grouped, k=2, dim=-1, largest=True, sorted=False).indices
    mask_grouped = torch.zeros_like(grouped)
    mask_grouped.scatter_(-1, top2, 1.0)
    return mask_grouped.view_as(weight)


def apply_2to4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = make_2to4_mask(weight)
    sparse_weight = weight * mask
    return sparse_weight, mask


def summarize(mask: torch.Tensor) -> dict[str, float]:
    zeros = (mask == 0).sum().item()
    total = mask.numel()
    sparsity = zeros / total
    return {
        "total_elements": float(total),
        "zero_elements": float(zeros),
        "sparsity": float(sparsity),
        "density": float(1.0 - sparsity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact 2:4 sparse weights from a dense tensor")
    parser.add_argument("--input", required=True, help="Path to .pt file containing a 2D tensor")
    parser.add_argument("--sparse-output", required=True, help="Path to save masked sparse tensor (.pt)")
    parser.add_argument("--mask-output", required=True, help="Path to save 0/1 mask tensor (.pt)")
    parser.add_argument("--report", default="", help="Optional JSON report path")
    args = parser.parse_args()

    weight = torch.load(args.input, map_location="cpu")
    if not isinstance(weight, torch.Tensor):
        raise ValueError("Expected torch.Tensor in input .pt file")

    sparse_weight, mask = apply_2to4(weight)

    Path(args.sparse_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.mask_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sparse_weight, args.sparse_output)
    torch.save(mask, args.mask_output)

    stats = summarize(mask)
    print(json.dumps(stats, indent=2))

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
