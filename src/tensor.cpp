#include "../include/tensor.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

Tensor::Tensor() {}

Tensor::Tensor(std::vector<size_t> shape) : shape(shape) {
    size_t total_size = 1;
    for (size_t dim : shape) total_size *= dim;
    
    data = std::make_shared<std::vector<float>>(total_size, 0.0f);
    grad = std::make_shared<std::vector<float>>(total_size, 0.0f);
    
    strides.resize(shape.size(), 1);
    size_t current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
}

size_t Tensor::size() const { return data->size(); }

void Tensor::print() const {
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\nData: [";
    for (size_t i = 0; i < data->size(); i++) {
        std::cout << (*data)[i];
        if (i < data->size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other) {
    // Case 1: Exact same shape
    if (shape == other->shape) {
        auto result = std::make_shared<Tensor>(shape);
        for (size_t i = 0; i < data->size(); i++) {
            (*result->data)[i] = (*data)[i] + (*other->data)[i];
        }
        result->left = shared_from_this();
        result->right = other;
        result->op = "add";
        return result;
    }
    
    // Case 2: Broadcasting a bias [1, N] to [M, N]
    if (shape.size() == 2 && other->shape.size() == 2 
        && shape[1] == other->shape[1] && other->shape[0] == 1) {
        
        size_t M = shape[0];
        size_t N = shape[1];
        auto result = std::make_shared<Tensor>(shape);
        
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                (*result->data)[i * N + j] = (*data)[i * N + j] + (*other->data)[j];
            }
        }
        
        result->left = shared_from_this();
        result->right = other;
        result->op = "add_broadcast";
        return result;
    }
    
    throw std::invalid_argument("Shapes not compatible for add.");
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) {
    if (shape.size() != 2 || other->shape.size() != 2) throw std::invalid_argument("Requires 2D.");
    if (shape[1] != other->shape[0]) throw std::invalid_argument("Dim mismatch.");
    size_t M = shape[0], K = shape[1], N = other->shape[1];
    auto result = std::make_shared<Tensor>(std::vector<size_t>{M, N});
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += (*data)[i * K + k] * (*other->data)[k * N + j];
            }
            (*result->data)[i * N + j] = sum;
        }
    }
    result->left = shared_from_this();
    result->right = other;
    result->op = "matmul";
    return result;
}

std::shared_ptr<Tensor> Tensor::relu() {
    auto result = std::make_shared<Tensor>(shape);
    for (size_t i = 0; i < data->size(); i++) {
        (*result->data)[i] = ((*data)[i] > 0.0f) ? (*data)[i] : 0.0f;
    }
    result->left = shared_from_this();
    result->op = "relu";
    return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
    auto result = std::make_shared<Tensor>(shape);
    for (size_t i = 0; i < data->size(); i++) {
        (*result->data)[i] = 1.0f / (1.0f + std::exp(-(*data)[i]));
    }
    result->left = shared_from_this();
    result->op = "sigmoid";
    return result;
}

void Tensor::backward() {
    if (op == "add") {
        for (size_t i = 0; i < grad->size(); i++) {
            (*left->grad)[i] += (*grad)[i];
            (*right->grad)[i] += (*grad)[i];
        }
    } 
    else if (op == "add_broadcast") {
        size_t M = left->shape[0];
        size_t N = left->shape[1];
        
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                (*left->grad)[i * N + j] += (*grad)[i * N + j];
            }
        }
        
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                (*right->grad)[j] += (*grad)[i * N + j];
            }
        }
    }
    else if (op == "matmul") {
        size_t M = left->shape[0], K = left->shape[1], N = right->shape[1];
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                float sum = 0.0f;
                for (size_t j = 0; j < N; j++) {
                    sum += (*grad)[i * N + j] * (*right->data)[k * N + j];
                }
                (*left->grad)[i * K + k] += sum;
            }
        }

        for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t i = 0; i < M; i++) {
                    sum += (*left->data)[i * K + k] * (*grad)[i * N + j];
                }
                (*right->grad)[k * N + j] += sum;
            }
        }
    }
    else if (op == "relu") {
        for (size_t i = 0; i < grad->size(); i++) {
            float g = ((*left->data)[i] > 0.0f) ? (*grad)[i] : 0.0f;
            (*left->grad)[i] += g;
        }
    }
    else if (op == "sigmoid") {
        for (size_t i = 0; i < grad->size(); i++) {
            float sig = (*data)[i]; 
            (*left->grad)[i] += (*grad)[i] * sig * (1.0f - sig);
        }
    }

    if (left) left->backward();
    if (right) right->backward();
}

float Tensor::mse_loss(std::shared_ptr<Tensor> target) {
    if (shape != target->shape) throw std::invalid_argument("Shape mismatch in MSE.");
    float loss = 0.0f;
    size_t N = data->size();
    for (size_t i = 0; i < N; i++) {
        float diff = (*data)[i] - (*target->data)[i];
        loss += diff * diff;
        (*grad)[i] = (2.0f / N) * diff;
    }
    return loss / N;
}

void Tensor::sgd_step(float learning_rate) {
    for (size_t i = 0; i < data->size(); i++) {
        (*data)[i] -= learning_rate * (*grad)[i];
    }
}

void Tensor::zero_grad() {
    std::fill(grad->begin(), grad->end(), 0.0f);
}
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
