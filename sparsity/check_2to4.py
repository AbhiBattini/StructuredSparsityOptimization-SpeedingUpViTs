"""Validate exact 2:4 compliance for a mask or sparse weight tensor."""

from __future__ import annotations

import argparse

import torch


def is_2to4_compliant(tensor: torch.Tensor, tol: float = 0.0) -> tuple[bool, int, int]:
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.shape}")

    rows, cols = tensor.shape
    if cols % 4 != 0:
        return False, rows * ((cols + 3) // 4), rows * ((cols + 3) // 4)

    nz = (tensor.abs() > tol).to(torch.int32)
    groups = nz.view(rows, cols // 4, 4)
    counts = groups.sum(dim=-1)
    valid = counts == 2
    invalid = (~valid).sum().item()
    total = valid.numel()
    return invalid == 0, int(total - invalid), int(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check exact 2:4 structure")
    parser.add_argument("--input", required=True, help="Path to .pt tensor")
    parser.add_argument("--tol", type=float, default=0.0, help="Absolute threshold for non-zero test")
    args = parser.parse_args()

    tensor = torch.load(args.input, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Expected tensor in input file")

    ok, good, total = is_2to4_compliant(tensor, tol=args.tol)
    ratio = good / total if total else 0.0
    print(f"2:4 groups valid: {good}/{total} ({ratio:.2%})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
