"""Run benchmark grids for dense, sparse, or production-simulation modes."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark grid")
    parser.add_argument("--mode", choices=["dense", "sparse", "production"], default="dense")
    parser.add_argument("--block-index", type=int, default=6)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "production":
        cmd = [
            sys.executable,
            "benchmarks/bench_production_workload.py",
            "--block-index",
            str(args.block_index),
            "--dtype",
            args.dtype,
            "--steps",
            str(args.steps),
        ]
    else:
        script = "benchmarks/bench_dense.py" if args.mode == "dense" else "benchmarks/bench_sparse.py"
        cmd = [
            sys.executable,
            script,
            "--block-index",
            str(args.block_index),
            "--dtype",
            args.dtype,
            "--batch-sizes",
            "1",
            "8",
            "32",
            "128",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
