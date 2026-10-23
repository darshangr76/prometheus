#include <iostream>
#include "../include/transformer_block.h"

int main() {
    size_t seq_len = 4;
    size_t embed_dim = 8;
    size_t num_heads = 2;
    size_t hidden_dim = 32;
    
    auto x = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    for (size_t i = 0; i < seq_len * embed_dim; i++) {
        (*x->data)[i] = (float)(i % 5) * 0.1f;
    }
    
    std::cout << "Input (4x8):" << std::endl;
    x->print();
    
    TransformerBlock block(embed_dim, num_heads, hidden_dim);
    auto out = block.forward(x, x, x, x);
    
    std::cout << "\n=== Transformer Block Output ===" << std::endl;
    out->print();
    
    return 0;
}