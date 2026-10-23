#include "../include/attention.h"
#include <cmath>

// We need a transpose function for the Key matrix (K^T)
std::shared_ptr<Tensor> transpose_2d(std::shared_ptr<Tensor> input) {
    size_t M = input->shape[0];
    size_t N = input->shape[1];
    
    auto result = std::make_shared<Tensor>(std::vector<size_t>{N, M});
    
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            (*result->data)[j * M + i] = (*input->data)[i * N + j];
        }
    }
    return result;
}

std::shared_ptr<Tensor> Attention::scaled_dot_product_attention(
    std::shared_ptr<Tensor> Q, 
    std::shared_ptr<Tensor> K, 
    std::shared_ptr<Tensor> V,
    bool causal_mask // New parameter
) {
    auto K_T = transpose_2d(K);
    auto scores = Q->matmul(K_T);
    
    float d_k = static_cast<float>(Q->shape[1]);
    float scale_factor = 1.0f / std::sqrt(d_k);
    size_t seq_len = Q->shape[0];
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (causal_mask && j > i) {
                // Mask out the future (set to negative infinity)
                (*scores->data)[i * seq_len + j] = -1e9f;
            } else {
                (*scores->data)[i * seq_len + j] *= scale_factor;
            }
        }
    }
    
    auto attention_weights = scores->softmax();
    auto output = attention_weights->matmul(V);
    
    return output;
}