#include <iostream>
#include "../include/ffn.h"

int main() {
    size_t seq_len = 3;
    size_t embed_dim = 4;
    size_t hidden_dim = 8;
    
    auto x = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    for (size_t i = 0; i < seq_len * embed_dim; i++) {
        (*x->data)[i] = (float)(i % 5) * 0.1f;
    }
    
    std::cout << "Input (3x4):" << std::endl;
    x->print();
    
    FeedForward ffn(embed_dim, hidden_dim);
    auto out = ffn.forward(x);
    
    std::cout << "\n=== FFN Output ===" << std::endl;
    out->print();
    
    return 0;
}