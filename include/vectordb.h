#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include "tensor.h"

struct VectorItem {
    std::string text;
    std::shared_ptr<Tensor> embedding;
    int id;
};

class VectorDB {
private:
    std::vector<VectorItem> items;
    int dim;

public:
    VectorDB(int dim);
    
    // Add a text and its embedding to the database
    void add(const std::string& text, std::shared_ptr<Tensor> embedding);
    
    // Search for the most similar text given a query embedding
    std::string search(std::shared_ptr<Tensor> query_embedding, int top_k = 1);
    
    // Calculate cosine similarity between two vectors
    float cosine_similarity(std::shared_ptr<Tensor> v1, std::shared_ptr<Tensor> v2);
};