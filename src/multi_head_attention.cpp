#include "../include/multi_head_attention.h"
#include <cmath>

MultiHeadAttention::MultiHeadAttention(int num_heads, int embed_dim) 
    : num_heads(num_heads), embed_dim(embed_dim) {
    // In a real transformer, we would check that embed_dim % num_heads == 0
    std::cout << "MultiHeadAttention initialized with " << num_heads 
              << " heads, embedding dim " << embed_dim << std::endl;
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(
    std::shared_ptr<Tensor> Q, 
    std::shared_ptr<Tensor> K, 
    std::shared_ptr<Tensor> V
) {
    // For simplicity, we are assuming a single batch and full embedding dim per head.
    // A real implementation would slice Q, K, V into num_heads chunks, 
    // run attention on each, and concatenate the results.
    
    // For this simplified version, we just call the single head attention
    // and return the result.
    return Attention::scaled_dot_product_attention(Q, K, V);
}