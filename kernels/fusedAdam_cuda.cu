#include <torch/extension.h>
#include <math.h> 
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
__global__ void fused_adam_cuda_impl(
    T* grad, T* m, T* v, int N, float beta, float gamma, float lr, float eps, float pow_beta, float pow_gamma
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        T grad_val = grad[tid];
        T m_val = m[tid];
        T v_val = v[tid];
        m_val = beta * m_val + (1 - beta) * grad_val;
        v_val = gamma * v_val + (1 - gamma) * grad_val * grad_val;
        float m_hat = m_val / (1 - pow_beta);
        float v_hat = v_val / (1 - pow_gamma);
        grad[tid] = - lr * m_hat / (sqrt(v_hat) + eps);
        m[tid] = m_val;
        v[tid] = v_val;
    }
}

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_REDUCED(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) 

#define AT_DISPATCH_FLOATING_TYPES_AND_REDUCED(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_REDUCED(__VA_ARGS__))


void fused_adam_cuda(
    torch::Tensor grad,
    torch::Tensor m,
    torch::Tensor v,
    float beta,
    float gamma,
    float lr,
    float eps,
    int step
) {
    int N = grad.numel();
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    float pow_beta = pow(beta, step);
    float pow_gamma = pow(gamma, step);
    AT_DISPATCH_FLOATING_TYPES_AND_REDUCED(grad.scalar_type(), "fused_adam", ([&] {
        fused_adam_cuda_impl<scalar_t><<<grid_size, block_size>>>(
            grad.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            N, beta, gamma, lr, eps, pow_beta, pow_gamma
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}