"""Benchmark dense vs sparse implementations for ViT MLP layer inference.

Compares:
- dense_original: original unpruned dense linear
- dense_masked: dense linear with 2:4-masked weights
- pytorch_semi: PyTorch semi-structured sparse tensor matmul
- custom_naive: custom CUDA naive sparse kernel
- custom_optimized: custom CUDA optimized sparse kernel
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

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


def _time_fn(fn, iters: int, warmup: int, device: torch.device) -> tuple[float, float, float]:
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


def run_dense(x3: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    y = torch.matmul(x3, w.t())
    if b is not None:
        y = y + b
    return y


def run_semistructured(
    x2: torch.Tensor,
    w_sparse,
    b: torch.Tensor | None,
    batch: int,
    seq_len: int,
) -> torch.Tensor:
    y2 = torch.mm(x2, w_sparse.t())
    if b is not None:
        y2 = y2 + b
    return y2.view(batch, seq_len, -1)


def _record(result_rows: list[Result], impl: str, bs: int, mean: float, p50: float, p95: float, err: float) -> None:
    thr = bs / (mean / 1000.0)
    result_rows.append(Result(impl=impl, batch=bs, mean_ms=mean, p50_ms=p50, p95_ms=p95, throughput=thr, max_error=err))


def main() -> None:
    p = argparse.ArgumentParser(description="Sparse benchmark (Weeks 3-6)")
    p.add_argument("--block-index", type=int, default=6)
    p.add_argument("--layer", choices=["fc1", "fc2"], default="fc1")
    p.add_argument("--seq-len", type=int, default=197)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--build-verbose", action="store_true")
    p.add_argument("--csv", default="", help="Optional CSV output path")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for sparse benchmark.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    weights = None if args.no_pretrained else ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).eval().to(device)
    layer = _select_layer(model, args.block_index, args.layer).to(device=device, dtype=dtype)

    with torch.no_grad():
        dense_orig_w = layer.weight.detach().clone()
        dense_b = layer.bias.detach().clone() if layer.bias is not None else None
        sparse_w, _ = apply_2to4(dense_orig_w)

    semi_ok = False
    w_semi = None
    try:
        from torch.sparse import to_sparse_semi_structured

        w_semi = to_sparse_semi_structured(sparse_w)
        semi_ok = True
    except Exception as exc:
        print(f"Semi-structured path unavailable: {exc}")

    custom_naive_ok = False
    custom_opt_ok = False
    packed = None
    try:
        packed = pack_2to4(sparse_w, dense_b)
        probe_x = torch.randn(4, sparse_w.size(1), device=device, dtype=dtype)
        _ = sparse_linear(probe_x, packed, impl="naive", verbose_build=args.build_verbose)
        custom_naive_ok = True
        _ = sparse_linear(probe_x, packed, impl="optimized", verbose_build=args.build_verbose)
        custom_opt_ok = True
    except Exception as exc:
        print(f"Custom CUDA path unavailable: {exc}")

    results: list[Result] = []
    print("impl\tbatch\tmean_ms\tp50_ms\tp95_ms\tthroughput(samples/s)\tmax_error")
    for bs in args.batch_sizes:
        x3 = torch.randn(bs, args.seq_len, layer.in_features, device=device, dtype=dtype)
        x2 = x3.view(-1, layer.in_features)

        y_ref_sparse = run_dense(x3, sparse_w, dense_b)
        y_ref_orig = run_dense(x3, dense_orig_w, dense_b)

        mean, p50, p95 = _time_fn(lambda: run_dense(x3, dense_orig_w, dense_b), args.iters, args.warmup, device)
        _record(results, "dense_original", bs, mean, p50, p95, 0.0)
        print(f"dense_original\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{bs/(mean/1000):.2f}\t0.0")

        mean, p50, p95 = _time_fn(lambda: run_dense(x3, sparse_w, dense_b), args.iters, args.warmup, device)
        err = (run_dense(x3, sparse_w, dense_b) - y_ref_sparse).abs().max().item()
        _record(results, "dense_masked", bs, mean, p50, p95, err)
        print(f"dense_masked\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{bs/(mean/1000):.2f}\t{err:.6f}")

        if semi_ok and w_semi is not None:
            out = run_semistructured(x2, w_semi, dense_b, bs, args.seq_len)
            err = (out - y_ref_sparse).abs().max().item()
            mean, p50, p95 = _time_fn(
                lambda: run_semistructured(x2, w_semi, dense_b, bs, args.seq_len),
                args.iters,
                args.warmup,
                device,
            )
            _record(results, "pytorch_semi", bs, mean, p50, p95, err)
            print(f"pytorch_semi\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{bs/(mean/1000):.2f}\t{err:.6f}")

        if custom_naive_ok and packed is not None:
            out2 = sparse_linear(x2, packed, impl="naive").view(bs, args.seq_len, -1)
            err2 = (out2 - y_ref_sparse).abs().max().item()
            mean, p50, p95 = _time_fn(
                lambda: sparse_linear(x2, packed, impl="naive").view(bs, args.seq_len, -1),
                args.iters,
                args.warmup,
                device,
            )
            _record(results, "custom_naive", bs, mean, p50, p95, err2)
            print(f"custom_naive\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{bs/(mean/1000):.2f}\t{err2:.6f}")

        if custom_opt_ok and packed is not None:
            out3 = sparse_linear(x2, packed, impl="optimized").view(bs, args.seq_len, -1)
            err3 = (out3 - y_ref_sparse).abs().max().item()
            mean, p50, p95 = _time_fn(
                lambda: sparse_linear(x2, packed, impl="optimized").view(bs, args.seq_len, -1),
                args.iters,
                args.warmup,
                device,
            )
            _record(results, "custom_optimized", bs, mean, p50, p95, err3)
            print(f"custom_optimized\t{bs}\t{mean:.4f}\t{p50:.4f}\t{p95:.4f}\t{bs/(mean/1000):.2f}\t{err3:.6f}")

        dense_delta = (y_ref_orig - y_ref_sparse).abs().max().item()
        if bs == args.batch_sizes[0]:
            print(f"note\tmax_diff_dense_original_vs_masked={dense_delta:.6f}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["impl", "batch", "mean_ms", "p50_ms", "p95_ms", "throughput", "max_error"])
            writer.writeheader()
            for r in results:
                writer.writerow(r.__dict__)
        print(f"Saved benchmark results to {path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
