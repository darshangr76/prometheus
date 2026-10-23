#include <iostream>
#include "../include/multi_head_attention.h"

int main() {
    size_t seq_len = 2;
    size_t embed_dim = 2;
    
    auto Q = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    auto K = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    auto V = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    
    (*Q->data) = {1.0f, 1.0f, 1.0f, 0.0f};
    (*K->data) = {1.0f, 0.0f, 0.0f, 1.0f};
    (*V->data) = {1.0f, 2.0f, 3.0f, 4.0f};
    
    MultiHeadAttention mha(2, embed_dim); // 2 heads
    auto output = mha.forward(Q, K, V);
    
    std::cout << "=== Multi-Head Attention Output ===" << std::endl;
    output->print();
    
    return 0;
}