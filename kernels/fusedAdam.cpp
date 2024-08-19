#include <torch/extension.h>
#include <iostream>
void matmul_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K);

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

void fused_adam_cuda(
    torch::Tensor grad,
    torch::Tensor m,
    torch::Tensor v,
    float beta,
    float gamma,
    float lr,
    float eps,
    int step
);
void fused_adam_cpu(
    torch::Tensor grad,
    torch::Tensor m,
    torch::Tensor v,
    float beta,
    float gamma,
    float lr,
    float eps,
    int step
);

void fused_adam(
    torch::Tensor grad,
    torch::Tensor m,
    torch::Tensor v,
    float beta,
    float gamma,
    float lr,
    float eps,
    int step
) {
    CHECK_INPUT(grad);
    CHECK_INPUT(m);
    CHECK_INPUT(v);

    if (grad.device().is_cuda()) {
        fused_adam_cuda(grad, m, v, beta, gamma, lr, eps, step);
    } else {
        fused_adam_cpu(grad, m, v, beta, gamma, lr, eps, step);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_adam", &fused_adam, "Fused Adam");
}