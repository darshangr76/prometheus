#pragma once
#include <memory>
#include "tensor.h"

class LayerNorm {
public:
    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    float eps = 1e-5f;

    LayerNorm(int embed_dim);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> d_out, std::shared_ptr<Tensor> x);
    void zero_grad();
    void sgd_step(float lr);
};