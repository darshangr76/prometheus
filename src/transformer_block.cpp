#include "../include/transformer_block.h"
#include <iostream>

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int hidden_dim)
    : mha(num_heads, embed_dim), 
      ffn(embed_dim, hidden_dim), 
      ln1(embed_dim), 
      ln2(embed_dim) {
    std::cout << "TransformerBlock created." << std::endl;
}

std::shared_ptr<Tensor> TransformerBlock::forward(
    std::shared_ptr<Tensor> x, 
    std::shared_ptr<Tensor> Q, 
    std::shared_ptr<Tensor> K, 
    std::shared_ptr<Tensor> V
) {
    // 1. Attention with Residual Connection
    auto norm_x = ln1.forward(x);
    auto attn_out = mha.forward(norm_x, norm_x, norm_x); // Self-attention uses the same input for Q, K, V
    auto x2 = x->add(attn_out);
    
    // 2. FFN with Residual Connection
    auto norm_x2 = ln2.forward(x2);
    auto ffn_out = ffn.forward(norm_x2);
    auto x3 = x2->add(ffn_out);
    
    return x3;
}