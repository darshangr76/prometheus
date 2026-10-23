#pragma once
#include <vector>
#include <random>
#include <iostream>

class Embedding {
public:
    int vocab_size;
    int embed_dim;
    std::vector<std::vector<float>> weights;

    Embedding(int vocab_size, int embed_dim);
    
    // Takes a list of token IDs, returns a 2D matrix of vectors
    std::vector<std::vector<float>> forward(const std::vector<int>& token_ids);
};