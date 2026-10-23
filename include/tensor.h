#pragma once
#include <vector>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#define USE_GPU true

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::shared_ptr<std::vector<float>> data;
    std::shared_ptr<std::vector<float>> grad;
    std::shared_ptr<Tensor> left;
    std::shared_ptr<Tensor> right;
    std::string op;

    // For Adam Optimizer (must be inside the class)
    std::shared_ptr<std::vector<float>> m;
    std::shared_ptr<std::vector<float>> v;

    Tensor();
    Tensor(std::vector<size_t> shape);
    
    void print() const;
    size_t size() const;

    // Math Operations
    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);
    
    // Activation Functions
    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> softmax();
    std::shared_ptr<Tensor> sigmoid();
    
    // Autograd
    void backward();
    
    // Loss Functions & Optimizers
    float mse_loss(std::shared_ptr<Tensor> target);
    void sgd_step(float learning_rate);
    void adam_step(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void zero_grad();
};