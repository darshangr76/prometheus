#include "../include/vectordb.h"
#include <algorithm>
#include <iostream>

VectorDB::VectorDB(int dim) : dim(dim) {}

float VectorDB::cosine_similarity(std::shared_ptr<Tensor> v1, std::shared_ptr<Tensor> v2) {
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot_product += (*v1->data)[i] * (*v2->data)[i];
        norm1 += (*v1->data)[i] * (*v1->data)[i];
        norm2 += (*v2->data)[i] * (*v2->data)[i];
    }
    
    if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

void VectorDB::add(const std::string& text, std::shared_ptr<Tensor> embedding) {
    VectorItem item;
    item.text = text;
    item.embedding = embedding;
    item.id = items.size();
    items.push_back(item);
}

std::string VectorDB::search(std::shared_ptr<Tensor> query_embedding, int top_k) {
    if (items.empty()) return "No items in DB.";
    
    // Calculate similarity for all items
    std::vector<std::pair<float, int>> scores;
    for (const auto& item : items) {
        float score = cosine_similarity(query_embedding, item.embedding);
        scores.push_back({score, item.id});
    }
    
    // Sort by score descending
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Return the top match (or combine top_k)
    std::string result = "";
    for (int i = 0; i < top_k && i < scores.size(); i++) {
        result += items[scores[i].second].text + "\n";
    }
    return result;
}