#include "../include/llm.h"

LLM::LLM(int num_layers, int embed_dim, int num_heads, int hidden_dim) : num_layers(num_layers) {
    for (int i = 0; i < num_layers; i++) {
        blocks.emplace_back(embed_dim, num_heads, hidden_dim);
    }
}

std::shared_ptr<Tensor> LLM::forward(std::shared_ptr<Tensor> x) {
    for (int i = 0; i < num_layers; i++) x = blocks[i].forward(x);
    return x;
}

std::shared_ptr<Tensor> LLM::backward(std::shared_ptr<Tensor> d_out) {
    for (int i = num_layers - 1; i >= 0; i--) d_out = blocks[i].backward(d_out);
    return d_out;;
}

void LLM::zero_grad() {
    for (auto& block : blocks) block.zero_grad();
}

void LLM::adam_step(float lr) {
    for (auto& block : blocks) block.adam_step(lr);
}
