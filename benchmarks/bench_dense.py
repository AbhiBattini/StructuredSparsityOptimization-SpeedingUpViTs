"""Benchmark dense forward latency for a target ViT-B/16 MLP linear layer.

Example:
    python benchmarks/bench_dense.py --batch-sizes 1 8 32 128 --dtype float16
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

try:
    from torchvision.models import ViT_B_16_Weights, vit_b_16
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torchvision is required. Install from requirements.txt") from exc


@dataclass
class BenchResult:
    batch_size: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_samples_per_s: float


def _select_layer(model: nn.Module, block_index: int, which: str) -> nn.Linear:
    block = model.encoder.layers[block_index]
    linears = [m for m in block.mlp if isinstance(m, nn.Linear)]
    if len(linears) != 2:
        raise ValueError(f"Expected 2 MLP linear layers, got {len(linears)}")
    return linears[0] if which == "fc1" else linears[1]


def _to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype in (torch.float16, torch.bfloat16):
        return x.to(dtype)
    return x


def benchmark_layer(
    layer: nn.Linear,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    device: torch.device,
) -> BenchResult:
    layer = layer.to(device=device)
    if dtype in (torch.float16, torch.bfloat16):
        layer = layer.to(dtype=dtype)

    x = torch.randn(batch_size, seq_len, layer.in_features, device=device)
    x = _to_dtype(x, dtype)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(warmup):
        _ = layer(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = layer(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
    throughput = batch_size / (mean / 1000.0)
    return BenchResult(batch_size, mean, p50, p95, throughput)


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense benchmark for ViT-B/16 MLP linear layer")
    parser.add_argument("--block-index", type=int, default=6)
    parser.add_argument("--layer", choices=["fc1", "fc2"], default="fc1")
    parser.add_argument("--seq-len", type=int, default=197, help="ViT-B/16 token count incl. CLS token")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        print("Warning: non-fp32 benchmark on CPU may be slow or unsupported; falling back to float32.")
        dtype = torch.float32

    weights = None if args.no_pretrained else ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).eval().to(device)
    layer = _select_layer(model, args.block_index, args.layer)

    print(
        f"Benchmarking block {args.block_index} MLP {args.layer} "
        f"({layer.in_features}->{layer.out_features}) on {device} with {dtype}"
    )
    print("batch\tmean_ms\tp50_ms\tp95_ms\tthroughput(samples/s)")

    for bs in args.batch_sizes:
        r = benchmark_layer(layer, bs, args.seq_len, dtype, args.warmup, args.iters, device)
        print(
            f"{r.batch_size}\t{r.latency_ms_mean:.4f}\t{r.latency_ms_p50:.4f}\t"
            f"{r.latency_ms_p95:.4f}\t{r.throughput_samples_per_s:.2f}"
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
