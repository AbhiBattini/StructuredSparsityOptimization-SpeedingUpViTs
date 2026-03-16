"""Python helpers for the naive CUDA 2:4 sparse linear kernel."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


@dataclass
class Sparse2to4Pack:
    values: torch.Tensor  # [N, G, 2]
    indices: torch.Tensor  # [N, G, 2], int32 offsets in [0,3]
    bias: torch.Tensor | None


ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    from torch.utils.cpp_extension import load

    return load(
        name="naive_sparse_ext",
        sources=[str(ROOT / "bindings.cpp"), str(ROOT / "naive_sparse_matmul.cu")],
        verbose=verbose,
        extra_cuda_cflags=["-O2"],
        extra_cflags=["-O2"],
    )


def pack_2to4(weight: torch.Tensor, bias: torch.Tensor | None = None) -> Sparse2to4Pack:
    """Pack a 2:4-compliant dense weight [N, K] into [N, G, 2] values/indices."""
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [N, K], got {weight.shape}")
    n, k = weight.shape
    if k % 4 != 0:
        raise ValueError("K must be divisible by 4")

    grouped = weight.view(n, k // 4, 4)
    nz_mask = grouped != 0
    nz_count = nz_mask.sum(dim=-1)
    if not torch.all(nz_count == 2):
        raise ValueError("weight is not exact 2:4-compliant (expected exactly 2 nonzeros/group)")

    idx = nz_mask.nonzero(as_tuple=False)
    values = torch.empty((n, k // 4, 2), dtype=weight.dtype, device=weight.device)
    indices = torch.empty((n, k // 4, 2), dtype=torch.int32, device=weight.device)

    # idx rows are [row, group, pos_in_4], we fill 2 entries per [row, group].
    cursor = torch.zeros((n, k // 4), dtype=torch.int64, device=weight.device)
    for r, g, p in idx:
        c = cursor[r, g].item()
        values[r, g, c] = grouped[r, g, p]
        indices[r, g, c] = p.to(torch.int32)
        cursor[r, g] += 1

    packed_bias = bias.contiguous() if bias is not None else None
    return Sparse2to4Pack(values.contiguous(), indices.contiguous(), packed_bias)


def sparse_linear(input_2d: torch.Tensor, packed: Sparse2to4Pack, verbose_build: bool = False) -> torch.Tensor:
    """Compute Y = X @ W^T + b using naive CUDA kernel.

    input_2d: [M, K]
    packed.values/indices correspond to W [N, K]
    returns: [M, N]
    """
    if not input_2d.is_cuda:
        raise ValueError("input_2d must be CUDA tensor")
    ext = load_extension(verbose=verbose_build)
    bias = packed.bias if packed.bias is not None else None
    return ext.naive_sparse_linear(
        input_2d.contiguous(),
        packed.values.contiguous(),
        packed.indices.contiguous(),
        bias,
    )
