#pragma once
#include <memory>
#include "tensor.h"

class FeedForward {
public:
    std::shared_ptr<Tensor> W1, B1, W2, B2;
    int embed_dim;
    int hidden_dim;

    FeedForward(int embed_dim, int hidden_dim);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};