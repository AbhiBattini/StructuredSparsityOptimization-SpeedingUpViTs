"""PyTorch semi-structured 2:4 sparse baseline skeleton.

This script is intentionally lightweight and checks whether the runtime supports
PyTorch's semi-structured sparse APIs before benchmarking.
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Check semi-structured sparse support in current PyTorch runtime")
    parser.parse_args()

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")

    try:
        from torch.sparse import to_sparse_semi_structured  # type: ignore

        print("Found torch.sparse.to_sparse_semi_structured")
        x = torch.randn(128, 128, device="cuda" if cuda_ok else "cpu", dtype=torch.float16 if cuda_ok else torch.float32)
        if x.shape[1] % 4 == 0:
            try:
                sx = to_sparse_semi_structured(x)
                print(f"Semi-structured tensor conversion succeeded: {type(sx)}")
            except Exception as exc:  # pragma: no cover
                print(f"Semi-structured conversion failed in this env: {exc}")
        else:
            print("Skipping conversion test: non-4-aligned shape")
    except Exception as exc:  # pragma: no cover
        print("Semi-structured APIs unavailable in this PyTorch build.")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
