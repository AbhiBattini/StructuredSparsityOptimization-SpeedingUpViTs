"""Simulate production-like request mix and compare implementations.

Workload model:
- bursty per-step request count sampled from a fixed pattern
- each request runs one target MLP forward with chosen batch size
- reports aggregate p50/p95 latency and throughput per implementation
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from benchmarks.bench_sparse import _select_layer, run_dense, run_semistructured
from kernels.naive_sparse import pack_2to4, sparse_linear
from sparsity.make_2to4 import apply_2to4
from torchvision.models import ViT_B_16_Weights, vit_b_16


def percentile(v: list[float], q: float) -> float:
    s = sorted(v)
    idx = max(0, min(len(s) - 1, int(q * len(s)) - 1))
    return s[idx]


def main() -> None:
    p = argparse.ArgumentParser(description="Production-like sparse inference simulation")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=197)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--block-index", type=int, default=6)
    p.add_argument("--layer", choices=["fc1", "fc2"], default="fc1")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval().to(device)
    layer = _select_layer(model, args.block_index, args.layer).to(device=device, dtype=dtype)

    dense_w = layer.weight.detach().clone()
    dense_b = layer.bias.detach().clone() if layer.bias is not None else None
    sparse_w, _ = apply_2to4(dense_w)
    packed = pack_2to4(sparse_w, dense_b)

    try:
        from torch.sparse import to_sparse_semi_structured

        w_semi = to_sparse_semi_structured(sparse_w)
        semi_ok = True
    except Exception:
        semi_ok = False
        w_semi = None

    impls = ["dense_original", "dense_masked", "custom_naive", "custom_optimized"]
    if semi_ok:
        impls.insert(2, "pytorch_semi")

    traffic_pattern = [1, 1, 2, 4, 8, 16, 8, 4, 2, 1, 32, 64, 32, 8, 2, 1]
    lat: dict[str, list[float]] = {k: [] for k in impls}
    total_samples: dict[str, int] = {k: 0 for k in impls}

    for step in range(args.steps):
        bs = traffic_pattern[step % len(traffic_pattern)]
        x3 = torch.randn(bs, args.seq_len, layer.in_features, device=device, dtype=dtype)
        x2 = x3.view(-1, layer.in_features)

        for impl in impls:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if impl == "dense_original":
                _ = run_dense(x3, dense_w, dense_b)
            elif impl == "dense_masked":
                _ = run_dense(x3, sparse_w, dense_b)
            elif impl == "pytorch_semi":
                _ = run_semistructured(x2, w_semi, dense_b, bs, args.seq_len)
            elif impl == "custom_naive":
                _ = sparse_linear(x2, packed, impl="naive").view(bs, args.seq_len, -1)
            elif impl == "custom_optimized":
                _ = sparse_linear(x2, packed, impl="optimized").view(bs, args.seq_len, -1)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            lat[impl].append((t1 - t0) * 1000)
            total_samples[impl] += bs

    print("impl\tp50_ms\tp95_ms\tavg_ms\tthroughput(samples/s)")
    for impl in impls:
        mean = statistics.mean(lat[impl])
        p50 = percentile(lat[impl], 0.50)
        p95 = percentile(lat[impl], 0.95)
        throughput = total_samples[impl] / (sum(lat[impl]) / 1000.0)
        print(f"{impl}\t{p50:.4f}\t{p95:.4f}\t{mean:.4f}\t{throughput:.2f}")


if __name__ == "__main__":
    main()
