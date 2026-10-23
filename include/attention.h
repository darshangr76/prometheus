#pragma once
#include <memory>
#include "tensor.h"

class Attention {
public:
    // The core attention function
    // Q, K, V are the Query, Key, and Value tensors
    static std::shared_ptr<Tensor> scaled_dot_product_attention(
        std::shared_ptr<Tensor> Q, 
        std::shared_ptr<Tensor> K, 
        std::shared_ptr<Tensor> V
    );
};