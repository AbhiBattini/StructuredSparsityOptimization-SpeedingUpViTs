# StructuredSparsityOptimization-SpeedingUpViTs

Benchmarking **when structured sparsity actually speeds up GPU inference** for ViT-B/16 MLP layers.

## 0) Get the repo (download/clone)
### Option A: clone with git
```bash
git clone https://github.com/<your-org>/StructuredSparsityOptimization-SpeedingUpViTs.git
cd StructuredSparsityOptimization-SpeedingUpViTs
```

### Option B: GitHub ZIP download
1. Open the repository page in GitHub.
2. Click **Code** → **Download ZIP**.
3. Extract ZIP and `cd` into the project folder.

## 1) Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Project scope
- Model: `torchvision.models.vit_b_16`
- Target layers: transformer-block MLP linear layers (`fc1` first, then `fc2`)
- Structured sparsity: exact 2:4 masks (50% sparsity)
- Core question: **which batch/shape/precision regimes favor sparse vs dense**?

## 3) Repository layout
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
  bindings.cpp
  naive_sparse_matmul.cu
  optimized_sparse_matmul.cu
  naive_sparse.py
benchmarks/
  bench_dense.py
  bench_sparse.py
  bench_grid.py
  bench_production_workload.py
  plot_results.py
  benchmark_report_template.md
profiling/
  nsight_notes.md
tests/
  test_correctness.py
```

## 4) Week-by-week deliverables (1-6)

### Week 1 — Baseline + model surgery
```bash
python models/download_vit_weights.py
python models/extract_vit_mlp.py --block-index 6
python benchmarks/bench_dense.py --batch-sizes 1 8 32 128 --dtype float16
```

### Week 2 — Exact 2:4 pipeline
```bash
python sparsity/make_2to4.py --input weight.pt --sparse-output sparse.pt --mask-output mask.pt --report report.json
python sparsity/check_2to4.py --input sparse.pt
```

### Week 3 + 4 — Sparse baselines + custom naive kernel
```bash
python benchmarks/bench_sparse.py --batch-sizes 1 8 32 128 --dtype float16 --csv benchmarks/results/sparse.csv
```
This benchmark compares:
- `dense_original`
- `dense_masked`
- `pytorch_semi` (if available)
- `custom_naive`
- `custom_optimized`

### Week 5 — optimization + profiling
- `optimized_sparse_matmul.cu` provides a first optimized kernel path.
- Profile instructions and metrics checklist are in:
```bash
cat profiling/nsight_notes.md
```

### Week 6 — polish + systems conclusions
```bash
python benchmarks/plot_results.py --csv benchmarks/results/sparse.csv --outdir benchmarks/plots
python benchmarks/bench_production_workload.py --steps 200 --dtype float16
```
Use:
- `benchmarks/benchmark_report_template.md`
- generated plots in `benchmarks/plots/`

## 5) Production workload simulation
`benchmarks/bench_production_workload.py` uses a bursty request pattern to mimic real inference demand, then reports p50/p95 latency and throughput for each implementation. This is where you can show whether sparse paths improve service-level behavior, not just microbenchmarks.

## 6) Notes on expected outcomes
A strong result is typically:
- Dense baseline remains hard to beat globally.
- PyTorch semi-structured sparse wins in some regimes.
- Custom naive kernel underperforms.
- Optimized kernel closes gap and may win in selected batch/shape windows.

## 7) Testing
```bash
python -m compileall models sparsity baselines benchmarks kernels tests
pytest -q
```
