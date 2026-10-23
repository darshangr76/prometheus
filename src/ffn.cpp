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
std::shared_ptr<Tensor> FeedForward::backward(std::shared_ptr<Tensor> d_out, std::shared_ptr<Tensor> x) {
    auto h1 = x->matmul(W1)->add(B1);
    auto h2 = h1->relu();
    
    size_t M = h2->shape[0], K = h2->shape[1], N = d_out->shape[1];
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t s = 0; s < M; s++) sum += (*h2->data)[s * K + i] * (*d_out->data)[s * N + j];
            (*W2->grad)[i * N + j] += sum;
        }
    }
    for (size_t j = 0; j < N; j++) {
        float sum = 0.0f;
        for (size_t s = 0; s < M; s++) sum += (*d_out->data)[s * N + j];
        (*B2->grad)[j] += sum;
    }
    
    auto d_h2 = std::make_shared<Tensor>(h2->shape);
    for (size_t s = 0; s < M; s++) {
        for (size_t i = 0; i < K; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < N; j++) sum += (*d_out->data)[s * N + j] * (*W2->data)[i * N + j];
            (*d_h2->data)[s * K + i] = sum;
        }
    }
    
    auto d_h1 = std::make_shared<Tensor>(h1->shape);
    for (size_t i = 0; i < h1->data->size(); i++) {
        (*d_h1->data)[i] = ((*h1->data)[i] > 0.0f) ? (*d_h2->data)[i] : 0.0f;
    }
    
    M = x->shape[0]; K = x->shape[1]; N = d_h1->shape[1];
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t s = 0; s < M; s++) sum += (*x->data)[s * K + i] * (*d_h1->data)[s * N + j];
            (*W1->grad)[i * N + j] += sum;
        }
    }
    for (size_t j = 0; j < N; j++) {
        float sum = 0.0f;
        for (size_t s = 0; s < M; s++) sum += (*d_h1->data)[s * N + j];
        (*B1->grad)[j] += sum;
    }
    
    auto d_x = std::make_shared<Tensor>(x->shape);
    for (size_t s = 0; s < M; s++) {
        for (size_t i = 0; i < K; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < N; j++) sum += (*d_h1->data)[s * N + j] * (*W1->data)[i * N + j];
            (*d_x->data)[s * K + i] = sum;
        }
    }
    return d_x;
}

void FeedForward::zero_grad() {
    W1->zero_grad(); B1->zero_grad(); W2->zero_grad(); B2->zero_grad();
}

void FeedForward::sgd_step(float lr) {
    W1->sgd_step(lr); B1->sgd_step(lr); W2->sgd_step(lr); B2->sgd_step(lr);
}
void FeedForward::adam_step(float lr) {
    W1->adam_step(lr); B1->adam_step(lr);
    W2->adam_step(lr); B2->adam_step(lr);
}
