#include <iostream>
#include "../include/vectordb.h"

int main() {
    int dim = 4;
    VectorDB db(dim);
    
    // 1. Create dummy embeddings (In a real system, these come from an embedding model)
    auto emb1 = std::make_shared<Tensor>(std::vector<size_t>{(size_t)dim});
    (*emb1->data) = {1.0f, 0.0f, 0.0f, 0.0f}; // Represents "apple"
    
    auto emb2 = std::make_shared<Tensor>(std::vector<size_t>{(size_t)dim});
    (*emb2->data) = {0.0f, 1.0f, 0.0f, 0.0f}; // Represents "banana"
    
    auto emb3 = std::make_shared<Tensor>(std::vector<size_t>{(size_t)dim});
    (*emb3->data) = {0.0f, 0.0f, 1.0f, 0.0f}; // Represents "car"
    
    // 2. Add to DB
    db.add("Apples are red fruits.", emb1);
    db.add("Bananas are yellow fruits.", emb2);
    db.add("Cars are fast vehicles.", emb3);
    
    std::cout << "VectorDB initialized with 3 items." << std::endl;
    
    // 3. Query
    auto query = std::make_shared<Tensor>(std::vector<size_t>{(size_t)dim});
    (*query->data) = {0.9f, 0.1f, 0.0f, 0.0f}; // Closest to "apple"
    
    std::cout << "Searching for: [0.9, 0.1, 0.0, 0.0]" << std::endl;
    std::string result = db.search(query);
    
    std::cout << "Closest match: " << result << std::endl;
    
    return 0;
}