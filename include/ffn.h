#pragma once
#include <memory>
#include "tensor.h"

class FeedForward {
public:
    std::shared_ptr<Tensor> W1, B1, W2, B2;
    int embed_dim, hidden_dim;
    FeedForward(int embed_dim, int hidden_dim);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> d_out, std::shared_ptr<Tensor> x);
    void zero_grad();
    void sgd_step(float lr);
};
