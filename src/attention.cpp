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
    std::shared_ptr<Tensor> V
) {
    // 1. Transpose K so we can multiply Q * K^T
    auto K_T = transpose_2d(K);
    
    // 2. Calculate Scores = Q @ K^T
    auto scores = Q->matmul(K_T);
    
    // 3. Scale the scores by 1 / sqrt(d_k)
    // d_k is the last dimension of Q (the embedding size)
    float d_k = static_cast<float>(Q->shape[1]);
    float scale_factor = 1.0f / std::sqrt(d_k);
    
    for (size_t i = 0; i < scores->data->size(); i++) {
        (*scores->data)[i] *= scale_factor;
    }
    
    // 4. Apply Softmax to get Attention Weights (probabilities)
    auto attention_weights = scores->softmax();
    
    // 5. Multiply the weights by V to get the final output
    auto output = attention_weights->matmul(V);
    
    return output;
}