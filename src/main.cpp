#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "../include/tensor.h"

int main() {
    // 1. Dataset
    auto X = std::make_shared<Tensor>(std::vector<size_t>{4, 2});
    (*X->data) = {0,0, 0,1, 1,0, 1,1};
    auto Y = std::make_shared<Tensor>(std::vector<size_t>{4, 1});
    (*Y->data) = {0, 1, 1, 0};

    // 2. Weights and Biases
    auto W1 = std::make_shared<Tensor>(std::vector<size_t>{2, 8});
    auto B1 = std::make_shared<Tensor>(std::vector<size_t>{1, 8});
    auto W2 = std::make_shared<Tensor>(std::vector<size_t>{8, 1});
    auto B2 = std::make_shared<Tensor>(std::vector<size_t>{1, 1});

    srand(time(NULL));
    auto randomize = [](std::shared_ptr<Tensor> t) {
        for (auto& val : *(t->data)) {
            val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    };
    randomize(W1); randomize(B1); randomize(W2); randomize(B2);

    float lr = 0.1f;

    // 3. Training Loop
    for (int epoch = 0; epoch <= 10000; epoch++) {
        W1->zero_grad(); B1->zero_grad();
        W2->zero_grad(); B2->zero_grad();

        // Forward pass with Broadcasting
        auto z1 = X->matmul(W1)->add(B1);
        auto a1 = z1->relu();
        auto z2 = a1->matmul(W2)->add(B2);
        auto pred = z2->sigmoid();

        float loss = pred->mse_loss(Y);
        pred->backward();

        W1->sgd_step(lr); B1->sgd_step(lr);
        W2->sgd_step(lr); B2->sgd_step(lr);

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss << std::endl;
        }
    }

    // 4. Test
    std::cout << "\n--- Final Predictions ---" << std::endl;
    auto final_pred = X->matmul(W1)->add(B1)->relu()->matmul(W2)->add(B2)->sigmoid();
    
    std::cout << "Input [0, 0] -> Pred: " << (*final_pred->data)[0] << " (Expected: 0)" << std::endl;
    std::cout << "Input [0, 1] -> Pred: " << (*final_pred->data)[1] << " (Expected: 1)" << std::endl;
    std::cout << "Input [1, 0] -> Pred: " << (*final_pred->data)[2] << " (Expected: 1)" << std::endl;
    std::cout << "Input [1, 1] -> Pred: " << (*final_pred->data)[3] << " (Expected: 0)" << std::endl;

    return 0;
}