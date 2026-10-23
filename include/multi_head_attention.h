#pragma once
#include <memory>
#include <vector>
#include "tensor.h"
#include "attention.h"

class MultiHeadAttention {
public:
    int num_heads;
    int embed_dim;
    
    MultiHeadAttention(int num_heads, int embed_dim);
    
    // Takes Q, K, V and returns the combined attention output
    std::shared_ptr<Tensor> forward(
        std::shared_ptr<Tensor> Q, 
        std::shared_ptr<Tensor> K, 
        std::shared_ptr<Tensor> V
    );
};