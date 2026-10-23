#include <iostream>
#include "../include/attention.h"

int main() {
    int embed_dim = 4;
    int seq_len = 3;
    
    // Create a dummy input tensor
    auto x = std::make_shared<Tensor>(std::vector<size_t>{(size_t)seq_len, (size_t)embed_dim});
    for (size_t i = 0; i < seq_len * embed_dim; i++) {
        (*x->data)[i] = (float)(i % 5) * 0.1f;
    }
    
    std::cout << "Input (3x4):" << std::endl;
    x->print();
    
    // Create the Attention layer
    Attention attn(embed_dim);
    
    // Forward pass
    auto out = attn.forward(x, false);
    std::cout << "\nAttention Output:" << std::endl;
    out->print();
    
    // Backward pass (just to check it runs)
    auto d_out = std::make_shared<Tensor>(out->shape);
    for (size_t i = 0; i < out->data->size(); i++) {
        (*d_out->data)[i] = 1.0f; // Fake gradient
    }
    
    auto d_x = attn.backward(d_out, false);
    std::cout << "\nBackward Output:" << std::endl;
    d_x->print();
    
    return 0;
}
