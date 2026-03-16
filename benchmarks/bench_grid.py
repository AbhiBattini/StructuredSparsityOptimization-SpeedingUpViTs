"""Run benchmark grid suggested by the project plan."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark grid for dense baseline")
    parser.add_argument("--block-index", type=int, default=6)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "benchmarks/bench_dense.py",
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
