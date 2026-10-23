#include <iostream>
#include <vector>
#include "../include/tokenizer.h"
#include "../include/embedding.h"
#include "../include/transformer_block.h"
#include "../include/lm_head.h"

int main() {
    int vocab_size = 96;
    int embed_dim = 16;
    int num_heads = 2;
    int hidden_dim = 64;
    
    Tokenizer tokenizer;
    Embedding embed(vocab_size, embed_dim);
    TransformerBlock block(embed_dim, num_heads, hidden_dim);
    LMHead lm_head(embed_dim, vocab_size);
    
    std::string training_text = "hello world! ";
    auto token_ids = tokenizer.encode(training_text);
    
    std::cout << "Training tokens: ";
    for (int id : token_ids) std::cout << id << " ";
    std::cout << std::endl;
    
    for (int epoch = 0; epoch < 100; epoch++) {
        // 1. Convert tokens to vectors, then to a Tensor
        auto embeddings_vec = embed.forward(token_ids);
        auto x = std::make_shared<Tensor>(std::vector<size_t>{embeddings_vec.size(), (size_t)embed_dim});
        for (size_t i = 0; i < embeddings_vec.size(); i++) {
            for (size_t j = 0; j < embed_dim; j++) {
                (*x->data)[i * embed_dim + j] = embeddings_vec[i][j];
            }
        }
        
        // 2. Transformer Block
        auto out = block.forward(x, x, x, x);
        
        // 3. Output Head
        auto logits = lm_head.forward(out);
        
        // 4. Targets (next token)
        std::vector<int> targets = token_ids;
        for (size_t i = 0; i < targets.size() - 1; i++) targets[i] = token_ids[i + 1];
        targets[targets.size() - 1] = token_ids[0];
        
        // 5. Loss
        float loss = lm_head.compute_loss(logits, targets);
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss << std::endl;
        }
    }
    
    return 0;
}