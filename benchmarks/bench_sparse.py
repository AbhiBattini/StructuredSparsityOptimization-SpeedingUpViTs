"""Benchmark dense vs PyTorch semi-structured vs naive custom CUDA sparse linear.

Example:
  python benchmarks/bench_sparse.py --batch-sizes 1 8 32 128 --dtype float16
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

from kernels.naive_sparse import pack_2to4, sparse_linear
from sparsity.make_2to4 import apply_2to4

try:
    from torchvision.models import ViT_B_16_Weights, vit_b_16
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torchvision is required. Install from requirements.txt") from exc


@dataclass
class Result:
    impl: str
    batch: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    throughput: float
    max_error: float


def _select_layer(model: nn.Module, block_index: int, which: str) -> nn.Linear:
    block = model.encoder.layers[block_index]
    linears = [m for m in block.mlp if isinstance(m, nn.Linear)]
    return linears[0] if which == "fc1" else linears[1]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_fn(fn, iters: int, warmup: int, device: torch.device):
    for _ in range(warmup):
        fn()
    _sync(device)
    lats = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000)
    mean = statistics.mean(lats)
    p50 = statistics.median(lats)
    p95 = sorted(lats)[max(0, int(0.95 * len(lats)) - 1)]
    return mean, p50, p95


def run_dense(x3: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None):
    y = torch.matmul(x3, w.t())
    if b is not None:
        y = y + b
    return y


def run_semistructured(x2: torch.Tensor, w_sparse, b: torch.Tensor | None, batch: int, seq_len: int):
    y2 = torch.mm(x2, w_sparse.t())
    if b is not None:
        y2 = y2 + b
    return y2.view(batch, seq_len, -1)


def main() -> None:
    p = argparse.ArgumentParser(description="Sparse benchmark (Week 3/4)")
    p.add_argument("--block-index", type=int, default=6)
    p.add_argument("--layer", choices=["fc1", "fc2"], default="fc1")
    p.add_argument("--seq-len", type=int, default=197)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--build-verbose", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for meaningful sparse benchmark.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    weights = None if args.no_pretrained else ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).eval().to(device)
    layer = _select_layer(model, args.block_index, args.layer).to(device=device, dtype=dtype)

    with torch.no_grad():
        dense_w = layer.weight.detach().clone()
        dense_b = layer.bias.detach().clone() if layer.bias is not None else None
        sparse_w, _ = apply_2to4(dense_w)

    try:
        from torch.sparse import to_sparse_semi_structured

        w_semi = to_sparse_semi_structured(sparse_w)
        semi_ok = True
    except Exception as exc:
        print(f"Semi-structured path unavailable: {exc}")
        semi_ok = False
        w_semi = None

    custom_ok = True
    try:
        packed = pack_2to4(sparse_w, dense_b)
        _ = sparse_linear(torch.randn(4, sparse_w.size(1), device=device, dtype=dtype), packed, verbose_build=args.build_verbose)
    except Exception as exc:
        print(f"Custom CUDA path unavailable: {exc}")
        custom_ok = False
        packed = None

    print("impl\tbatch\tmean_ms\tp50_ms\tp95_ms\tthroughput(samples/s)\tmax_error")
    for bs in args.batch_sizes:
        x3 = torch.randn(bs, args.seq_len, layer.in_features, device=device, dtype=dtype)
        x2 = x3.view(-1, layer.in_features)

        y_ref = run_dense(x3, sparse_w, dense_b)

        # dense (masked) baseline
        mean, p50, p95 = _time_fn(lambda: run_dense(x3, sparse_w, dense_b), args.iters, args.warmup, device)
        thr = bs / (mean / 1000.0)
        print(f"dense_masked\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{thr:.2f}\t0.0")

        if semi_ok and w_semi is not None:
            out = run_semistructured(x2, w_semi, dense_b, bs, args.seq_len)
            err = (out - y_ref).abs().max().item()
            mean, p50, p95 = _time_fn(
                lambda: run_semistructured(x2, w_semi, dense_b, bs, args.seq_len),
                args.iters,
                args.warmup,
                device,
            )
            thr = bs / (mean / 1000.0)
            print(f"pytorch_semi\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{thr:.2f}\t{err:.6f}")

        if custom_ok and packed is not None:
            out2 = sparse_linear(x2, packed).view(bs, args.seq_len, -1)
            err2 = (out2 - y_ref).abs().max().item()
            mean, p50, p95 = _time_fn(
                lambda: sparse_linear(x2, packed).view(bs, args.seq_len, -1),
                args.iters,
                args.warmup,
                device,
            )
            thr = bs / (mean / 1000.0)
            print(f"custom_naive\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{thr:.2f}\t{err2:.6f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
