"""Python helpers for custom CUDA 2:4 sparse linear kernels."""

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
        name="sparse_ext",
        sources=[
            str(ROOT / "bindings.cpp"),
            str(ROOT / "naive_sparse_matmul.cu"),
            str(ROOT / "optimized_sparse_matmul.cu"),
        ],
        verbose=verbose,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
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

    indices = nz_mask.to(torch.int32).argsort(dim=-1, descending=True)[..., :2]
    indices, _ = torch.sort(indices, dim=-1)
    values = torch.gather(grouped, dim=-1, index=indices.to(torch.int64))

    packed_bias = bias.contiguous() if bias is not None else None
    return Sparse2to4Pack(values.contiguous(), indices.contiguous(), packed_bias)


def sparse_linear(
    input_2d: torch.Tensor,
    packed: Sparse2to4Pack,
    impl: str = "naive",
    verbose_build: bool = False,
) -> torch.Tensor:
    """Compute Y = X @ W^T + b using custom CUDA kernel.

    impl: "naive" | "optimized"
    """
    if not input_2d.is_cuda:
        raise ValueError("input_2d must be CUDA tensor")
    ext = load_extension(verbose=verbose_build)
    bias = packed.bias if packed.bias is not None else None

    if impl == "naive":
        fn = ext.naive_sparse_linear
    elif impl == "optimized":
        fn = ext.optimized_sparse_linear
    else:
        raise ValueError(f"Unknown impl={impl}; expected 'naive' or 'optimized'")

    return fn(
        input_2d.contiguous(),
        packed.values.contiguous(),
        packed.indices.contiguous(),
        bias,
    )
