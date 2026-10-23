#include "../include/attention.h"
#include <cmath>
#include <random>
#include <iostream>

Attention::Attention(int embed_dim) : embed_dim(embed_dim) {
    W_q = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)embed_dim});
    W_k = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)embed_dim});
    W_v = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)embed_dim});
    W_o = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)embed_dim});
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& val : *(W_q->data)) val = dist(gen);
    for (auto& val : *(W_k->data)) val = dist(gen);
    for (auto& val : *(W_v->data)) val = dist(gen);
    for (auto& val : *(W_o->data)) val = dist(gen);
}

std::shared_ptr<Tensor> Attention::forward(std::shared_ptr<Tensor> x, bool causal_mask) {
    saved_input = x;
    saved_Q = x->matmul(W_q);
    saved_K = x->matmul(W_k);
    saved_V = x->matmul(W_v);
    
    auto K_T = std::make_shared<Tensor>(std::vector<size_t>{saved_K->shape[1], saved_K->shape[0]});
    for (size_t i = 0; i < saved_K->shape[0]; i++)
        for (size_t j = 0; j < saved_K->shape[1]; j++)
            (*K_T->data)[j * saved_K->shape[0] + i] = (*saved_K->data)[i * saved_K->shape[1] + j];
            
    auto scores = saved_Q->matmul(K_T);
    float d_k = (float)saved_Q->shape[1];
    float scale = 1.0f / std::sqrt(d_k);
    size_t seq_len = saved_Q->shape[0];
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (causal_mask && j > i) (*scores->data)[i * seq_len + j] = -1e9f;
            else (*scores->data)[i * seq_len + j] *= scale;
        }
    }
    
    saved_attn_weights = scores->softmax();
    auto context = saved_attn_weights->matmul(saved_V);
    return context->matmul(W_o);
}

std::shared_ptr<Tensor> Attention::backward(std::shared_ptr<Tensor> d_out, bool causal_mask) {
    size_t seq_len = saved_Q->shape[0];
    size_t d_k = saved_Q->shape[1];
    
    // 1. d_context = d_out @ W_o^T
    auto W_o_T = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)embed_dim});
    for (size_t i = 0; i < embed_dim; i++)
        for (size_t j = 0; j < embed_dim; j++)
            (*W_o_T->data)[j * embed_dim + i] = (*W_o->data)[i * embed_dim + j];
    auto d_context = d_out->matmul(W_o_T);
    
    // 2. d_W_o = context^T @ d_out
    auto context = saved_attn_weights->matmul(saved_V);
    auto context_T = std::make_shared<Tensor>(std::vector<size_t>{context->shape[1], context->shape[0]});
    for (size_t i = 0; i < context->shape[0]; i++)
        for (size_t j = 0; j < context->shape[1]; j++)
            (*context_T->data)[j * context->shape[0] + i] = (*context->data)[i * context->shape[1] + j];
    
    auto d_W_o = context_T->matmul(d_out);
    for (size_t i = 0; i < d_W_o->data->size(); i++) {
        (*W_o->grad)[i] += (*d_W_o->data)[i];
    }
    
    // 3. d_V = attention_weights^T @ d_context
    auto attn_T = std::make_shared<Tensor>(std::vector<size_t>{saved_attn_weights->shape[1], saved_attn_weights->shape[0]});
    for (size_t i = 0; i < saved_attn_weights->shape[0]; i++)
        for (size_t j = 0; j < saved_attn_weights->shape[1]; j++)
            (*attn_T->data)[j * saved_attn_weights->shape[0] + i] = (*saved_attn_weights->data)[i * saved_attn_weights->shape[1] + j];
    auto d_V = attn_T->matmul(d_context);
    
    // 4. d_attn = d_context @ V^T
    auto V_T = std::make_shared<Tensor>(std::vector<size_t>{saved_V->shape[1], saved_V->shape[0]});
    for (size_t i = 0; i < saved_V->shape[0]; i++)
        for (size_t j = 0; j < saved_V->shape[1]; j++)
            (*V_T->data)[j * saved_V->shape[0] + i] = (*saved_V->data)[i * saved_V->shape[1] + j];
    auto d_attn = d_context->matmul(V_T);
    
    // 5. Softmax backward
    auto d_scores = std::make_shared<Tensor>(saved_attn_weights->shape);
    for (size_t i = 0; i < seq_len; i++) {
        float sum_d_attn = 0.0f;
        for (size_t j = 0; j < seq_len; j++) {
            if (causal_mask && j > i) continue;
            sum_d_attn += (*d_attn->data)[i * seq_len + j] * (*saved_attn_weights->data)[i * seq_len + j];
        }
        for (size_t j = 0; j < seq_len; j++) {
            if (causal_mask && j > i) (*d_scores->data)[i * seq_len + j] = 0.0f;
            else (*d_scores->data)[i * seq_len + j] = (*saved_attn_weights->data)[i * seq_len + j] * ((*d_attn->data)[i * seq_len + j] - sum_d_attn);
        }
    }
    
    // 6. Scale backward
    float scale = 1.0f / std::sqrt((float)d_k);
    for (size_t i = 0; i < d_scores->data->size(); i++) (*d_scores->data)[i] *= scale;
    
    // 7. d_Q = d_scores @ K, d_K = d_scores^T @ Q
    auto d_Q = d_scores->matmul(saved_K);
    auto d_scores_T = std::make_shared<Tensor>(std::vector<size_t>{d_scores->shape[1], d_scores->shape[0]});
    for (size_t i = 0; i < d_scores->shape[0]; i++)
        for (size_t j = 0; j < d_scores->shape[1]; j++)
            (*d_scores_T->data)[j * d_scores->shape[0] + i] = (*d_scores->data)[i * d_scores->shape[1] + j];
    auto d_K = d_scores_T->matmul(saved_Q);
    
    // 8. d_x = d_Q @ W_q^T
    auto W_q_T = std::make_shared<Tensor>(std::vector<size_t>{(size_t)embed_dim, (size_t)embed_dim});
    for (size_t i = 0; i < embed_dim; i++)
        for (size_t j = 0; j < embed_dim; j++)
            (*W_q_T->data)[j * embed_dim + i] = (*W_q->data)[i * embed_dim + j];
    auto d_x = d_Q->matmul(W_q_T);
    
    // 9. Accumulate weight gradients
    auto input_T = std::make_shared<Tensor>(std::vector<size_t>{saved_input->shape[1], saved_input->shape[0]});
    for (size_t i = 0; i < saved_input->shape[0]; i++)
        for (size_t j = 0; j < saved_input->shape[1]; j++)
            (*input_T->data)[j * saved_input->shape[0] + i] = (*saved_input->data)[i * saved_input->shape[1] + j];
    
    auto d_W_q = input_T->matmul(d_Q);
    auto d_W_k = input_T->matmul(d_K);
    auto d_W_v = input_T->matmul(d_V);
    
    for (size_t i = 0; i < d_W_q->data->size(); i++) {
        (*W_q->grad)[i] += (*d_W_q->data)[i];
        (*W_k->grad)[i] += (*d_W_k->data)[i];
        (*W_v->grad)[i] += (*d_W_v->data)[i];
    }
    
    return d_x;
}

void Attention::zero_grad() {
    W_q->zero_grad(); W_k->zero_grad(); W_v->zero_grad(); W_o->zero_grad();
}

void Attention::sgd_step(float lr) {
    W_q->sgd_step(lr); W_k->sgd_step(lr); W_v->sgd_step(lr); W_o->sgd_step(lr);
}

void Attention::adam_step(float lr) {
    W_q->adam_step(lr); W_k->adam_step(lr); W_v->adam_step(lr); W_o->adam_step(lr);
}
