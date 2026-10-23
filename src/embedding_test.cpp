#include <iostream>
#include "../include/embedding.h"

int main() {
    int vocab_size = 96;
    int embed_dim = 4; // Small for testing, real models use 768 or 4096
    
    Embedding embed(vocab_size, embed_dim);
    
    // Let's use the first 3 characters of "Hello" (H, e, l)
    std::vector<int> token_ids = {40, 69, 76};
    
    std::cout << "Token IDs: 40, 69, 76" << std::endl;
    std::cout << "Embedding Dimension: " << embed_dim << "\n" << std::endl;
    
    auto vectors = embed.forward(token_ids);
    
    for (size_t i = 0; i < vectors.size(); i++) {
        std::cout << "Token " << token_ids[i] << " -> [ ";
        for (float val : vectors[i]) {
            std::cout << val << " ";
        }
        std::cout << "]" << std::endl;
    }
    
    return 0;
}