#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

class Tokenizer {
private:
    std::unordered_map<char, int> char_to_id;
    std::unordered_map<int, char> id_to_char;
    int vocab_size;

public:
    Tokenizer();
    
    // Convert text to a list of integer IDs
    std::vector<int> encode(const std::string& text);
    
    // Convert a list of IDs back to text
    std::string decode(const std::vector<int>& ids);
    
    int get_vocab_size() const { return vocab_size; }
};