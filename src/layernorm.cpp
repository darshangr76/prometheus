#include "../include/layernorm.h"
#include <cmath>
#include <iostream>

LayerNorm::LayerNorm(int embed_dim) {
    gamma = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim});
    beta = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim});
    
    // Initialize gamma to 1 and beta to 0
    for (auto& val : *(gamma->data)) val = 1.0f;
    for (auto& val : *(beta->data)) val = 0.0f;
    
    std::cout << "LayerNorm initialized for dim " << embed_dim << std::endl;
}

std::shared_ptr<Tensor> LayerNorm::forward(std::shared_ptr<Tensor> x) {
    size_t rows = x->shape[0];
    size_t cols = x->shape[1];
    
    auto result = std::make_shared<Tensor>(x->shape);
    
    for (size_t i = 0; i < rows; i++) {
        // 1. Calculate mean for this row
        float mean = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            mean += (*x->data)[i * cols + j];
        }
        mean /= cols;
        
        // 2. Calculate variance
        float var = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float diff = (*x->data)[i * cols + j] - mean;
            var += diff * diff;
        }
        var /= cols;
        
        // 3. Normalize
        float std_dev = std::sqrt(var + eps);
        for (size_t j = 0; j < cols; j++) {
            float normalized = ((*x->data)[i * cols + j] - mean) / std_dev;
            // 4. Apply gamma and beta
            (*result->data)[i * cols + j] = normalized * (*gamma->data)[j] + (*beta->data)[j];
        }
    }
    
    return result;
}