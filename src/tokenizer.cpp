#include "../include/tokenizer.h"

Tokenizer::Tokenizer() {
    // We will use a simple ASCII vocabulary for now.
    // In a real LLM, this is replaced by BPE (Byte Pair Encoding) which combines common letters.
    
    // We map every printable ASCII character (32 to 126) to a unique ID.
    // Plus we add a special token for newlines.
    int id = 0;
    for (char c = 32; c <= 126; c++) {
        char_to_id[c] = id;
        id_to_char[id] = c;
        id++;
    }
    // Add newline
    char_to_id['\n'] = id;
    id_to_char[id] = '\n';
    id++;
    
    vocab_size = id;
    std::cout << "Tokenizer initialized. Vocab size: " << vocab_size << std::endl;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    for (char c : text) {
        if (char_to_id.find(c) != char_to_id.end()) {
            ids.push_back(char_to_id[c]);
        } else {
            // If character is unknown, map to space
            ids.push_back(char_to_id[' ']);
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string text = "";
    for (int id : ids) {
        if (id_to_char.find(id) != id_to_char.end()) {
            text += id_to_char[id];
        }
    }
    return text;
}
std::shared_ptr<Tensor> Tensor::softmax() {
    auto result = std::make_shared<Tensor>(shape);
    
    // Assuming 2D tensor for attention (batch/seq, features)
    // We apply softmax across each row (last dimension)
    size_t rows = shape[0];
    size_t cols = shape[1];
    
    for (size_t i = 0; i < rows; i++) {
        // 1. Find max value for this row (numerical stability)
        float max_val = (*data)[i * cols];
        for (size_t j = 1; j < cols; j++) {
            if ((*data)[i * cols + j] > max_val) {
                max_val = (*data)[i * cols + j];
            }
        }
        
        // 2. Calculate exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float exp_val = std::exp((*data)[i * cols + j] - max_val);
            (*result->data)[i * cols + j] = exp_val;
            sum += exp_val;
        }
        
        // 3. Normalize
        for (size_t j = 0; j < cols; j++) {
            (*result->data)[i * cols + j] /= sum;
        }
    }
    
    result->left = shared_from_this();
    result->op = "softmax";
    return result;
}
