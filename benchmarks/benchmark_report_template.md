# Structured sparsity benchmark report template (Week 6)

## Setup
- GPU:
- CUDA version:
- PyTorch version:
- TorchVision version:
- Precision: fp16/bf16/fp32
- Target layer: ViT-B/16 block __, `mlp_fc1` or `mlp_fc2`

## Dense baseline (Week 1)
| Batch | Mean latency (ms) | Throughput (samples/s) |
|---:|---:|---:|
| 1 | | |
| 8 | | |
| 32 | | |
| 128 | | |

## 2:4 pruning validation (Week 2)
- Exact 2:4 compliance:
- Sparsity ratio:
- Max deviation from compliance checker:

## Sparse benchmark matrix (Weeks 3-5)
| Impl | Batch | Mean latency (ms) | p95 (ms) | Throughput (samples/s) | Max error |
|---|---:|---:|---:|---:|---:|
| dense_original | | | | | |
| dense_masked | | | | | |
| pytorch_semi | | | | | |
| custom_naive | | | | | |
| custom_optimized | | | | | |

## Production workload simulation
| Impl | p50 latency (ms) | p95 latency (ms) | Avg latency (ms) | Throughput (samples/s) |
|---|---:|---:|---:|---:|

## Findings
- Where sparse wins:
- Where dense still wins:
- Break-even batch sizes:
- Why (bandwidth-bound vs compute-bound):
