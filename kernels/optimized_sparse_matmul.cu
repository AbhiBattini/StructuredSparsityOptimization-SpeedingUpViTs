#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void optimized_sparse_linear_kernel(
    const scalar_t* __restrict__ x,       // [M, K]
    const scalar_t* __restrict__ values,  // [N, G, 2]
    const int32_t* __restrict__ idx,      // [N, G, 2]
    const scalar_t* __restrict__ bias,    // [N] or nullptr
    scalar_t* __restrict__ y,             // [M, N]
    int M,
    int K,
    int N,
    int G) {
  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;

  const int local_n = threadIdx.x;
  const int local_m = threadIdx.y;
  const int n = blockIdx.x * TILE_N + local_n;
  const int m = blockIdx.y * TILE_M + local_m;

  if (m >= M || n >= N) {
    return;
  }

  const scalar_t* x_row = x + static_cast<int64_t>(m) * K;
  const int64_t w_base = static_cast<int64_t>(n) * G * 2;

  float acc = 0.0f;
#pragma unroll 4
  for (int g = 0; g < G; ++g) {
    const int64_t off = w_base + static_cast<int64_t>(g) * 2;
    const int p0 = idx[off];
    const int p1 = idx[off + 1];
    const int k0 = g * 4 + p0;
    const int k1 = g * 4 + p1;
    const float xv0 = static_cast<float>(x_row[k0]);
    const float xv1 = static_cast<float>(x_row[k1]);
    const float wv0 = static_cast<float>(values[off]);
    const float wv1 = static_cast<float>(values[off + 1]);
    acc = fmaf(xv0, wv0, acc);
    acc = fmaf(xv1, wv1, acc);
  }

  if (bias != nullptr) {
    acc += static_cast<float>(bias[n]);
  }
  y[static_cast<int64_t>(m) * N + n] = static_cast<scalar_t>(acc);
}

}  // namespace

at::Tensor optimized_sparse_linear_cuda(
    const at::Tensor& input,
    const at::Tensor& values,
    const at::Tensor& indices,
    c10::optional<at::Tensor> bias) {
  const auto M = static_cast<int>(input.size(0));
  const auto N = static_cast<int>(values.size(0));
  const auto G = static_cast<int>(values.size(1));
  const auto K = static_cast<int>(input.size(1));

  auto y = at::zeros({M, N}, input.options());

  const dim3 threads(16, 16);
  const dim3 blocks((N + 15) / 16, (M + 15) / 16);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "optimized_sparse_linear_cuda", [&] {
    const auto* bias_ptr = bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr;
    optimized_sparse_linear_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        values.data_ptr<scalar_t>(),
        indices.data_ptr<int32_t>(),
        bias_ptr,
        y.data_ptr<scalar_t>(),
        M,
        K,
        N,
        G);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
