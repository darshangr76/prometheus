#include "../include/transformer_block.h"

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int hidden_dim)
    : mha(num_heads, embed_dim), 
      ffn(embed_dim, hidden_dim), 
      ln1(embed_dim), 
      ln2(embed_dim) {
}

std::shared_ptr<Tensor> TransformerBlock::forward(std::shared_ptr<Tensor> x) {
    saved_x = x;
    
    auto norm_x = ln1.forward(x);
    saved_norm_x = norm_x;
    
    auto attn_out = mha.forward(norm_x);
    saved_attn_out = attn_out;
    
    auto x2 = x->add(attn_out);
    saved_x2 = x2;
    
    auto norm_x2 = ln2.forward(x2);
    saved_norm_x2 = norm_x2;
    
    auto ffn_out = ffn.forward(norm_x2);
    saved_ffn_out = ffn_out;
    
    return x2->add(ffn_out);
}

std::shared_ptr<Tensor> TransformerBlock::backward(std::shared_ptr<Tensor> d_out) {
    auto d_ffn_out = d_out;
    auto d_x2 = d_out;
    
    auto d_norm_x2 = ffn.backward(d_ffn_out, saved_norm_x2);
    
    auto d_x2_from_ln2 = ln2.backward(d_norm_x2, saved_x2);
    for (size_t i = 0; i < d_x2->data->size(); i++) {
        (*d_x2->data)[i] += (*d_x2_from_ln2->data)[i];
    }
    
    auto d_attn_out = d_x2;
    auto d_x = d_x2;
    
    auto d_norm_x = mha.backward(d_attn_out);
    
    auto d_x_from_ln1 = ln1.backward(d_norm_x, saved_x);
    for (size_t i = 0; i < d_x->data->size(); i++) {
        (*d_x->data)[i] += (*d_x_from_ln1->data)[i];
    }
    
    return d_x;
}
