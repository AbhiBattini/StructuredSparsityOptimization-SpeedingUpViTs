"""Plot latency and throughput from benchmark CSV produced by bench_sparse.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Plot sparse benchmark results")
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default="benchmarks/plots")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig1 = plt.figure(figsize=(8, 5))
    for impl, g in df.groupby("impl"):
        g = g.sort_values("batch")
        plt.plot(g["batch"], g["mean_ms"], marker="o", label=impl)
    plt.xlabel("Batch size")
    plt.ylabel("Mean latency (ms)")
    plt.title("Latency vs batch size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(outdir / "latency_vs_batch.png", dpi=160)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 5))
    for impl, g in df.groupby("impl"):
        g = g.sort_values("batch")
        plt.plot(g["batch"], g["throughput"], marker="o", label=impl)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (samples/s)")
    plt.title("Throughput vs batch size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outdir / "throughput_vs_batch.png", dpi=160)
    plt.close(fig2)

    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    main()
