#include <torch/extension.h>

at::Tensor naive_sparse_linear_cuda(
    const at::Tensor& input,
    const at::Tensor& values,
    const at::Tensor& indices,
    c10::optional<at::Tensor> bias);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor naive_sparse_linear(
    const at::Tensor& input,
    const at::Tensor& values,
    const at::Tensor& indices,
    c10::optional<at::Tensor> bias) {
  CHECK_CUDA(input);
  CHECK_CUDA(values);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(values);
  CHECK_CONTIGUOUS(indices);
  if (bias.has_value()) {
    CHECK_CUDA(bias.value());
    CHECK_CONTIGUOUS(bias.value());
  }
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
  TORCH_CHECK(values.dim() == 3, "values must be 3D [N, G, 2]");
  TORCH_CHECK(indices.dim() == 3, "indices must be 3D [N, G, 2]");

  const auto k = input.size(1);
  TORCH_CHECK(k % 4 == 0, "K must be divisible by 4");
  TORCH_CHECK(values.size(1) == k / 4, "values G dimension must match K/4");
  TORCH_CHECK(indices.size(1) == k / 4, "indices G dimension must match K/4");
  TORCH_CHECK(values.size(2) == 2, "values last dim must be 2");
  TORCH_CHECK(indices.size(2) == 2, "indices last dim must be 2");

  return naive_sparse_linear_cuda(input, values, indices, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_sparse_linear", &naive_sparse_linear, "Naive 2:4 sparse linear forward (CUDA)");
}
