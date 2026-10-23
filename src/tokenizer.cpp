#include "../include/tokenizer.h"

Tokenizer::Tokenizer() {
    int id = 0;
    for (char c = 32; c <= 126; c++) {
        char_to_id[c] = id;
        id_to_char[id] = c;
        id++;
    }
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
