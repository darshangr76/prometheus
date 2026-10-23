#include <iostream>
#include "../include/tokenizer.h"

int main() {
    Tokenizer tokenizer;
    
    std::string text = "Hello, Prometheus!";
    std::cout << "Original text: " << text << std::endl;
    
    // 1. Encode
    std::vector<int> ids = tokenizer.encode(text);
    std::cout << "Encoded IDs: ";
    for (int id : ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    // 2. Decode
    std::string decoded = tokenizer.decode(ids);
    std::cout << "Decoded text: " << decoded << std::endl;
    
    return 0;
}