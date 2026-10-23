#include "../include/ffn.h"
#include <random>
#include <iostream>

FeedForward::FeedForward(int embed_dim, int hidden_dim) 
    : embed_dim(embed_dim), hidden_dim(hidden_dim) {
    
    W1 = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)hidden_dim});
    B1 = std::make_shared<Tensor>(std::vector<size_t>{1, (size_t)hidden_dim});
    W2 = std::make_shared<Tensor>(std::vector<size_t>{(size_t)hidden_dim, (size_t)embed_dim});
    B2 = std::make_shared<Tensor>(std::vector<size_t>{1, (size_t)embed_dim});
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    for (auto& val : *(W1->data)) val = dist(gen);
    for (auto& val : *(B1->data)) val = 0.0f;
    for (auto& val : *(W2->data)) val = dist(gen);
    for (auto& val : *(B2->data)) val = 0.0f;
    
    std::cout << "FeedForward initialized: " << embed_dim << " -> " << hidden_dim << " -> " << embed_dim << std::endl;
}

std::shared_ptr<Tensor> FeedForward::forward(std::shared_ptr<Tensor> x) {
    // 1. Linear 1 + Bias: x @ W1 + B1
    auto h1 = x->matmul(W1)->add(B1);
    
    // 2. ReLU
    auto h2 = h1->relu();
    
    // 3. Linear 2 + Bias: h2 @ W2 + B2
    auto out = h2->matmul(W2)->add(B2);
    
    return out;
}