# Nsight profiling notes (Week 5)

## Target kernels
- `naive_sparse_linear`
- `optimized_sparse_linear`

## Profiling workflow
1. Build extension once using:
   ```bash
   python -c "from kernels.naive_sparse import load_extension; load_extension(verbose=True)"
   ```
2. Run a representative sparse benchmark:
   ```bash
   python benchmarks/bench_sparse.py --batch-sizes 1 8 32 128 --dtype float16 --iters 200 --warmup 50 --csv benchmarks/results/sparse.csv
   ```
3. Collect Nsight Systems timeline:
   ```bash
   nsys profile -o profiling/nsys_sparse_report python benchmarks/bench_sparse.py --batch-sizes 32 --dtype float16 --iters 200 --warmup 50
   ```
4. Collect Nsight Compute kernel metrics:
   ```bash
   ncu --set full -o profiling/ncu_sparse_report python benchmarks/bench_sparse.py --batch-sizes 32 --dtype float16 --iters 50 --warmup 20
   ```

## Metrics to track
- Kernel duration (ms)
- Achieved occupancy (%)
- DRAM throughput (GB/s)
- SM throughput (% peak)
- Warp branch efficiency
- FLOP utilization vs memory stalls

## Optimization hypotheses
- Naive kernel likely memory-bound with poor cache reuse.
- Optimized kernel should improve arithmetic efficiency with `fmaf` and unrolling.
- Further gains expected from shared-memory tiling and better metadata packing in Week 5+ iterations.

## Artifact checklist
- Save profiler screenshots under `profiling/artifacts/`.
- Include before/after table in final report:
  - `naive` vs `optimized`
  - occupancy
  - kernel time
  - throughput
