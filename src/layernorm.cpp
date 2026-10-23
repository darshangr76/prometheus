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
std::shared_ptr<Tensor> LayerNorm::backward(std::shared_ptr<Tensor> d_out, std::shared_ptr<Tensor> x) {
    size_t rows = x->shape[0], cols = x->shape[1];
    auto d_x = std::make_shared<Tensor>(x->shape);
    for (size_t i = 0; i < rows; i++) {
        float mean = 0.0f;
        for (size_t j = 0; j < cols; j++) mean += (*x->data)[i * cols + j];
        mean /= cols;
        float var = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float diff = (*x->data)[i * cols + j] - mean;
            var += diff * diff;
        }
        var /= cols;
        float std_dev = std::sqrt(var + eps);
        float sum_d_out = 0.0f, sum_d_out_x_hat = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float x_hat = ((*x->data)[i * cols + j] - mean) / std_dev;
            float d_out_val = (*d_out->data)[i * cols + j] * (*gamma->data)[j];
            sum_d_out += d_out_val;
            sum_d_out_x_hat += d_out_val * x_hat;
        }
        for (size_t j = 0; j < cols; j++) {
            float x_hat = ((*x->data)[i * cols + j] - mean) / std_dev;
            float d_out_val = (*d_out->data)[i * cols + j] * (*gamma->data)[j];
            (*d_x->data)[i * cols + j] = (1.0f / cols) * (1.0f / std_dev) * (cols * d_out_val - sum_d_out - x_hat * sum_d_out_x_hat);
            (*gamma->grad)[j] += (*d_out->data)[i * cols + j] * x_hat;
            (*beta->grad)[j] += (*d_out->data)[i * cols + j];
        }
    }
    return d_x;
}

void LayerNorm::zero_grad() {
    gamma->zero_grad();
    beta->zero_grad();
}

void LayerNorm::sgd_step(float lr) {
    gamma->sgd_step(lr);
    beta->sgd_step(lr);
}
void LayerNorm::adam_step(float lr) {
    gamma->adam_step(lr);
    beta->adam_step(lr);
}
