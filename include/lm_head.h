#pragma once
#include <memory>
#include <vector>
#include "tensor.h"

class LMHead {
public:
    std::shared_ptr<Tensor> weights;
    int embed_dim;
    int vocab_size;

    LMHead(int embed_dim, int vocab_size);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    float compute_loss(std::shared_ptr<Tensor> logits, const std::vector<int>& targets);
    void simple_update(std::shared_ptr<Tensor> x, const std::vector<int>& targets, float lr);
};