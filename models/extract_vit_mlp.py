"""Utilities to locate and inspect ViT-B/16 MLP linear layers.

Usage:
    python models/extract_vit_mlp.py --block-index 6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn

try:
    from torchvision.models import ViT_B_16_Weights, vit_b_16
except ImportError as exc:  # pragma: no cover - import guard for user envs
    raise SystemExit(
        "torchvision is required. Install dependencies from requirements.txt"
    ) from exc


@dataclass
class LayerInfo:
    block_index: int
    layer_name: str
    in_features: int
    out_features: int
    bias: bool
    params: int


def _linear_info(block_index: int, layer_name: str, linear: nn.Linear) -> LayerInfo:
    return LayerInfo(
        block_index=block_index,
        layer_name=layer_name,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        params=linear.weight.numel() + (linear.bias.numel() if linear.bias is not None else 0),
    )


def extract_mlp_linear_layers(model: nn.Module) -> list[LayerInfo]:
    layers: list[LayerInfo] = []
    for block_index, block in enumerate(model.encoder.layers):
        mlp_linears = [m for m in block.mlp if isinstance(m, nn.Linear)]
        if len(mlp_linears) != 2:
            raise ValueError(
                f"Expected exactly 2 Linear layers in MLP for block {block_index}, got {len(mlp_linears)}"
            )
        layers.append(_linear_info(block_index, "mlp_fc1", mlp_linears[0]))
        layers.append(_linear_info(block_index, "mlp_fc2", mlp_linears[1]))
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ViT-B/16 MLP linear layers")
    parser.add_argument(
        "--block-index",
        type=int,
        default=6,
        help="Transformer block index to highlight as primary benchmark target",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use randomly initialized model instead of pretrained weights",
    )
    parser.add_argument(
        "--json",
        default="",
        help="Optional output path for a JSON dump of all MLP linear layer metadata",
    )
    args = parser.parse_args()

    weights = None if args.no_pretrained else ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.eval()

    infos = extract_mlp_linear_layers(model)

    print("ViT-B/16 MLP Linear Layers")
    print("=" * 48)
    for info in infos:
        marker = " <-- primary target" if info.block_index == args.block_index and info.layer_name == "mlp_fc1" else ""
        print(
            f"block {info.block_index:02d} | {info.layer_name:7s} | "
            f"{info.in_features:4d} -> {info.out_features:4d} | "
            f"params={info.params:,}{marker}"
        )

    target = [i for i in infos if i.block_index == args.block_index and i.layer_name == "mlp_fc1"]
    if target:
        t = target[0]
        print("\nRecommended primary benchmark layer:")
        print(
            f"block {t.block_index} {t.layer_name} with shape "
            f"[{t.out_features}, {t.in_features}] (GEMM-friendly expansion layer)."
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(i) for i in infos], f, indent=2)
        print(f"\nSaved metadata to {args.json}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
