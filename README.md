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
  download_vit_weights.py
sparsity/
  make_2to4.py
  check_2to4.py
baselines/
  dense_pytorch.py
  pytorch_semistructured.py
kernels/
  naive_sparse_matmul.cu
  naive_sparse.py
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
2. Download ViT-B/16 pretrained weights into cache:
   ```bash
   python models/download_vit_weights.py
   ```
3. Inspect ViT MLP layer shapes:
   ```bash
   python models/extract_vit_mlp.py --block-index 6
   ```
4. Run dense baseline benchmark:
   ```bash
   python benchmarks/bench_dense.py --batch-sizes 1 8 32 128 --dtype float16
   ```
5. Run Week 3/4 sparse benchmark (dense masked vs PyTorch semi-structured vs custom naive CUDA):
   ```bash
   python benchmarks/bench_sparse.py --batch-sizes 1 8 32 128 --dtype float16
   ```
6. Build and verify 2:4 sparsity mask from a saved weight tensor:
   ```bash
   python sparsity/make_2to4.py --input weight.pt --sparse-output sparse.pt --mask-output mask.pt --report report.json
   python sparsity/check_2to4.py --input sparse.pt
   ```

## Current status
- ✅ Week 1: model surgery + dense benchmarking harness.
- ✅ Week 2: exact 2:4 mask generation and compliance checks.
- ✅ Week 3: PyTorch semi-structured sparse benchmark integration.
- ✅ Week 4: naive custom CUDA sparse matmul kernel + Python bindings + benchmark integration.
- 🔜 Next: kernel optimization, Nsight-driven tuning, end-to-end layer replacement.

## Notes
- Custom kernel expects exact 2:4-compliant weights and currently targets straightforward readability over performance.
- `benchmarks/bench_sparse.py` computes `max_error` against a dense reference using the same masked weights.
