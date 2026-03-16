"""Run Week 3 PyTorch semi-structured sparse benchmark path.

Example:
  python baselines/pytorch_semistructured.py --batch-sizes 1 8 32 128 --dtype float16
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.bench_sparse import main as bench_sparse_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-only", action="store_true", help="Only validate API/runtime support")
    args, _ = parser.parse_known_args()

    if args.check_only:
        print(f"CUDA available: {torch.cuda.is_available()}")
        try:
            from torch.sparse import to_sparse_semi_structured  # noqa: F401

            print("Found torch.sparse.to_sparse_semi_structured")
        except Exception as exc:
            print(f"Semi-structured APIs unavailable: {exc}")
        return

    bench_sparse_main()


if __name__ == "__main__":
    main()
