#include "../include/embedding.h"

Embedding::Embedding(int vocab_size, int embed_dim) 
    : vocab_size(vocab_size), embed_dim(embed_dim) {
    
    // Initialize the weights with random numbers
    // In a real LLM, these weights are learned during training
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f); // Mean 0, StdDev 1

    weights.resize(vocab_size, std::vector<float>(embed_dim));
    
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embed_dim; j++) {
            weights[i][j] = dist(gen);
        }
    }
}

std::vector<std::vector<float>> Embedding::forward(const std::vector<int>& token_ids) {
    std::vector<std::vector<float>> output;
    output.reserve(token_ids.size());
    
    for (int id : token_ids) {
        if (id >= 0 && id < vocab_size) {
            output.push_back(weights[id]);
        } else {
            // If unknown ID, return a vector of zeros
            output.push_back(std::vector<float>(embed_dim, 0.0f));
        }
    }
    
    return output;
}