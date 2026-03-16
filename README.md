# StructuredSparsityOptimization-SpeedingUpViTs

This repo benchmarks when **structured sparsity** (especially NVIDIA-friendly **2:4 sparsity**) actually improves GPU inference latency for ViT MLP layers.

## Scope
- Model: `torchvision.models.vit_b_16`
- Primary target: MLP linear layers in transformer blocks (start with middle-block `fc1` expansion layer)
- Core question: when does sparse inference beat dense inference in realistic workload settings?

## Repository layout
```text
models/
  extract_vit_mlp.py
sparsity/
  make_2to4.py
  check_2to4.py
baselines/
  dense_pytorch.py
  pytorch_semistructured.py
kernels/
  naive_sparse_matmul.cu
  optimized_sparse_matmul.cu
  bindings.cpp
benchmarks/
  bench_dense.py
  bench_sparse.py
  bench_grid.py
profiling/
  nsight_notes.md
tests/
  test_correctness.py
```

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Inspect ViT MLP layer shapes:
   ```bash
   python models/extract_vit_mlp.py --block-index 6
   ```
3. Run dense baseline benchmark:
   ```bash
   python benchmarks/bench_dense.py --batch-sizes 1 8 32 128 --dtype float16
   ```
4. Build and verify 2:4 sparsity mask from a saved weight tensor:
   ```bash
   python sparsity/make_2to4.py --input weight.pt --sparse-output sparse.pt --mask-output mask.pt --report report.json
   python sparsity/check_2to4.py --input sparse.pt
   ```

## Current status (foundation)
- ✅ Week 1 foundations: model surgery + dense benchmarking harness.
- ✅ Week 2 foundations: exact 2:4 mask generation and compliance checks.
- 🔜 Next: PyTorch semi-structured sparse benchmarking and custom CUDA kernels.

## Benchmark guidance
Use this base matrix:
- batch sizes: `1, 8, 32, 128`
- dtype: `fp16` on CUDA-capable GPUs
- implementations:
  - dense PyTorch
  - PyTorch semi-structured sparse
  - custom naive sparse kernel
  - custom optimized sparse kernel
- metrics:
  - latency
  - throughput
  - max error vs dense
  - optional memory footprint
