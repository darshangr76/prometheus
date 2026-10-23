#include <iostream>
#include <vector>
#include "../include/tokenizer.h"
#include "../include/embedding.h"
#include "../include/transformer_block.h"
#include "../include/lm_head.h"

int main() {
    int vocab_size = 96;
    int embed_dim = 32; // Increased from 16
    int num_heads = 4;  // Increased from 2
    int hidden_dim = 128; // Increased from 64
    
    Tokenizer tokenizer;
    Embedding embed(vocab_size, embed_dim);
    TransformerBlock block(embed_dim, num_heads, hidden_dim);
    LMHead lm_head(embed_dim, vocab_size);
    
    auto token_ids = tokenizer.encode("hello world! ");
    float lr = 0.1f; // Increased from 0.01
    
    for (int epoch = 0; epoch < 5000; epoch++) { // Increased from 500
        lm_head.weights->zero_grad();
        block.zero_grad();
        
        auto embeddings_vec = embed.forward(token_ids);
        auto x = std::make_shared<Tensor>(std::vector<size_t>{embeddings_vec.size(), (size_t)embed_dim});
        for (size_t i = 0; i < embeddings_vec.size(); i++)
            for (size_t j = 0; j < embed_dim; j++)
                (*x->data)[i * embed_dim + j] = embeddings_vec[i][j];
        
        auto out = block.forward(x);
        auto logits = lm_head.forward(out);
        
        std::vector<int> targets = token_ids;
        for (size_t i = 0; i < targets.size() - 1; i++) targets[i] = token_ids[i + 1];
        targets[targets.size() - 1] = token_ids[0];
        
        float loss = lm_head.compute_loss(logits, targets);
        
        if (epoch % 500 == 0) std::cout << "Epoch " << epoch << " | Loss: " << loss << std::endl;
        
        auto d_out = lm_head.backward(out, logits, targets);
        block.backward(d_out);
        
        lm_head.weights->sgd_step(lr);
        block.sgd_step(lr);
    }
    
    // Testing Generation
    std::cout << "\n--- Testing Generation ---" << std::endl;
    std::vector<int> prompt = {72, 69, 76, 76}; // "hell"
    auto prompt_vec = embed.forward(prompt);
    auto prompt_x = std::make_shared<Tensor>(std::vector<size_t>{prompt_vec.size(), (size_t)embed_dim});
    for (size_t i = 0; i < prompt_vec.size(); i++)
        for (size_t j = 0; j < embed_dim; j++)
            (*prompt_x->data)[i * embed_dim + j] = prompt_vec[i][j];
    
    auto prompt_out = block.forward(prompt_x);
    auto prompt_logits = lm_head.forward(prompt_out);
    
    size_t last_idx = prompt.size() - 1;
    int best_token = 0;
    float best_val = (*prompt_logits->data)[last_idx * vocab_size];
    for (int j = 1; j < vocab_size; j++) {
        if ((*prompt_logits->data)[last_idx * vocab_size + j] > best_val) {
            best_val = (*prompt_logits->data)[last_idx * vocab_size + j];
            best_token = j;
        }
    }
    
    std::cout << "Prompt: 'hell'" << std::endl;
    std::cout << "Predicted next token: '" << (char)best_token << "' (ID: " << best_token << ")" << std::endl;
    std::cout << "Full word: 'hell" << (char)best_token << "'" << std::endl;
    
    return 0;
}