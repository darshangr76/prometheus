#include "../include/lm_head.h"
#include <cmath>
#include <random>

LMHead::LMHead(int embed_dim, int vocab_size) 
    : embed_dim(embed_dim), vocab_size(vocab_size) {
    weights = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)vocab_size});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& val : *(weights->data)) val = dist(gen);
}

std::shared_ptr<Tensor> LMHead::forward(std::shared_ptr<Tensor> x) {
    return x->matmul(weights);
}

float LMHead::compute_loss(std::shared_ptr<Tensor> logits, const std::vector<int>& targets) {
    float loss = 0.0f;
    size_t seq_len = logits->shape[0];
    for (size_t i = 0; i < seq_len; i++) {
        float max_val = (*logits->data)[i * vocab_size];
        for (size_t j = 1; j < vocab_size; j++) {
            if ((*logits->data)[i * vocab_size + j] > max_val) 
                max_val = (*logits->data)[i * vocab_size + j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j < vocab_size; j++) {
            sum += std::exp((*logits->data)[i * vocab_size + j] - max_val);
        }
        int target = targets[i];
        float target_prob = std::exp((*logits->data)[i * vocab_size + target] - max_val) / sum;
        loss += -std::log(target_prob + 1e-10f);
    }
    return loss / seq_len;
}

void LMHead::simple_update(std::shared_ptr<Tensor> x, const std::vector<int>& targets, float lr) {
    for (size_t i = 0; i < x->shape[0]; i++) {
        int target = targets[i];
        for (size_t j = 0; j < embed_dim; j++) {
            (*weights->data)[j * vocab_size + target] += lr * (*x->data)[i * embed_dim + j];
        }
    }
}

std::shared_ptr<Tensor> LMHead::backward(std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> logits, const std::vector<int>& targets) {
    auto d_logits = std::make_shared<Tensor>(logits->shape);
    size_t seq_len = logits->shape[0];
    
    for (size_t i = 0; i < seq_len; i++) {
        float max_val = (*logits->data)[i * vocab_size];
        for (size_t j = 1; j < vocab_size; j++) {
            if ((*logits->data)[i * vocab_size + j] > max_val) 
                max_val = (*logits->data)[i * vocab_size + j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j < vocab_size; j++) {
            sum += std::exp((*logits->data)[i * vocab_size + j] - max_val);
        }
        for (size_t j = 0; j < vocab_size; j++) {
            float prob = std::exp((*logits->data)[i * vocab_size + j] - max_val) / sum;
            float target_val = (j == targets[i]) ? 1.0f : 0.0f;
            (*d_logits->data)[i * vocab_size + j] = prob - target_val;
        }
    }
    
    size_t embed_dim = weights->shape[0];
    auto d_weights = std::make_shared<Tensor>(weights->shape);
    for (size_t i = 0; i < embed_dim; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            float sum = 0.0f;
            for (size_t s = 0; s < seq_len; s++) {
                sum += (*out->data)[s * embed_dim + i] * (*d_logits->data)[s * vocab_size + j];
            }
            (*d_weights->data)[i * vocab_size + j] = sum;
        }
    }
    
    for (size_t i = 0; i < weights->data->size(); i++) {
        (*weights->grad)[i] += (*d_weights->data)[i];
    }
    
    auto d_out = std::make_shared<Tensor>(out->shape);
    for (size_t s = 0; s < seq_len; s++) {
        for (size_t i = 0; i < embed_dim; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < vocab_size; j++) {
                sum += (*d_logits->data)[s * vocab_size + j] * (*weights->data)[i * vocab_size + j];
            }
            (*d_out->data)[s * embed_dim + i] = sum;
        }
    }
    return d_out;
}
