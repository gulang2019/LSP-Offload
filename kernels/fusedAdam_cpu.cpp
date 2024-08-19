#include <torch/extension.h>
#include <math.h>
#include <immintrin.h>
#include <iostream>

void fused_adam_cpu_impl(float* grads, float* exp_avg, float* exp_avg_sq,
                     const float lr, const float beta1, const float beta2, const float eps,
                    const float pow_beta1, const float pow_beta2,
                     const size_t length) {
    // Assuming length is a multiple of 8 for simplicity
#pragma omp parallel for
    for (int i = 0; i < length; i += 8) {
        // Load elements
        __m256 v_grads = _mm256_loadu_ps(grads + i);
        __m256 v_exp_avg = _mm256_loadu_ps(exp_avg + i);
        __m256 v_exp_avg_sq = _mm256_loadu_ps(exp_avg_sq + i);

        // Update biased first moment estimate
        v_exp_avg = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(beta1), v_exp_avg),
                                  _mm256_mul_ps(_mm256_set1_ps(1.0f - beta1), v_grads));

        // Update biased second raw moment estimate
        v_exp_avg_sq = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(beta2), v_exp_avg_sq),
                                     _mm256_mul_ps(_mm256_set1_ps(1.0f - beta2), _mm256_mul_ps(v_grads, v_grads)));

        // Compute bias-corrected first moment estimate
        __m256 v_exp_avg_corr = _mm256_div_ps(v_exp_avg, _mm256_set1_ps(1.0f - pow_beta1));

        // Compute bias-corrected second raw moment estimate
        __m256 v_exp_avg_sq_corr = _mm256_div_ps(v_exp_avg_sq, _mm256_set1_ps(1.0f - pow_beta2));

        // Update parameters
        __m256 v_denom = _mm256_sqrt_ps(_mm256_add_ps(v_exp_avg_sq_corr, _mm256_set1_ps(eps)));
        v_grads = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(_mm256_set1_ps(lr), _mm256_div_ps(v_exp_avg_corr, v_denom)));

        // Store back to memory
        _mm256_storeu_ps(grads + i, v_grads);
        _mm256_storeu_ps(exp_avg + i, v_exp_avg);
        _mm256_storeu_ps(exp_avg_sq + i, v_exp_avg_sq);
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

void fused_adam_cpu(
    torch::Tensor grad,
    torch::Tensor m,
    torch::Tensor v,
    float beta,
    float gamma,
    float lr,
    float eps,
    int step
) {
    auto grad_data = grad.data_ptr<float>();
    auto m_data = m.data_ptr<float>();
    auto v_data = v.data_ptr<float>();
    int size = grad.numel();
    const float pow_beta1 = pow(beta, step);
    const float pow_beta2 = pow(gamma, step);

    if (size & 0x7) {
        std::cerr << "Input size must be a multiple of 8" << std::endl;
        return;
    }
    
    AT_DISPATCH_FLOATING_TYPES_AND_REDUCED(grad.scalar_type(), "fused_adam_impl", [&] {
        fused_adam_cpu_impl(grad_data, m_data, v_data, lr, beta, gamma, eps, pow_beta1, pow_beta2, size);
    });
}
