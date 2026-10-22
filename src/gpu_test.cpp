#include <iostream>

extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K);

int main() {
    int M = 2, N = 3, K = 2;
    float h_A[4] = {1, 2, 3, 4}; 
    float h_B[6] = {5, 6, 7, 8, 9, 10};
    float h_C[6] = {0};

    matmul_gpu(h_A, h_B, h_C, M, N, K);

    std::cout << "GPU Result:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}