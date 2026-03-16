#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void naive_sparse_linear_kernel(
    const scalar_t* __restrict__ x,       // [M, K]
    const scalar_t* __restrict__ values,  // [N, G, 2]
    const int32_t* __restrict__ idx,      // [N, G, 2], values in [0, 3]
    const scalar_t* __restrict__ bias,    // [N] or nullptr
    scalar_t* __restrict__ y,             // [M, N]
    int M,
    int K,
    int N,
    int G) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (n >= N || m >= M) {
    return;
  }

  const scalar_t* x_row = x + m * K;
  const int64_t base = static_cast<int64_t>(n) * G * 2;

  float acc = 0.0f;
  for (int g = 0; g < G; ++g) {
    const int64_t off = base + static_cast<int64_t>(g) * 2;
    const int k0 = g * 4 + idx[off];
    const int k1 = g * 4 + idx[off + 1];
    acc += static_cast<float>(values[off]) * static_cast<float>(x_row[k0]);
    acc += static_cast<float>(values[off + 1]) * static_cast<float>(x_row[k1]);
  }
  if (bias != nullptr) {
    acc += static_cast<float>(bias[n]);
  }
  y[m * N + n] = static_cast<scalar_t>(acc);
}

}  // namespace

at::Tensor naive_sparse_linear_cuda(
    const at::Tensor& input,
    const at::Tensor& values,
    const at::Tensor& indices,
    c10::optional<at::Tensor> bias) {
  const auto M = static_cast<int>(input.size(0));
  const auto K = static_cast<int>(input.size(1));
  const auto N = static_cast<int>(values.size(0));
  const auto G = static_cast<int>(values.size(1));

  auto y = at::zeros({M, N}, input.options());

  const dim3 threads(16, 16);
  const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "naive_sparse_linear_cuda", [&] {
    const auto* bias_ptr = bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr;
    naive_sparse_linear_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
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
