#include <iostream>
#include "../include/attention.h"

int main() {
    size_t seq_len = 2;
    size_t embed_dim = 2;
    
    auto Q = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    auto K = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    auto V = std::make_shared<Tensor>(std::vector<size_t>{seq_len, embed_dim});
    
    // Q = [[1, 1], [1, 0]]
    (*Q->data) = {1.0f, 1.0f, 1.0f, 0.0f};
    // K = [[1, 0], [0, 1]]
    (*K->data) = {1.0f, 0.0f, 0.0f, 1.0f};
    // V = [[1, 2], [3, 4]]
    (*V->data) = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto output = Attention::scaled_dot_product_attention(Q, K, V);
    
    std::cout << "=== Attention Output ===" << std::endl;
    output->print();
    
    return 0;
}