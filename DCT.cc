#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstdlib>
using namespace std;

void DCT_2D_Parallel(int N, float *input, float *output) {
    float alpha_u, alpha_v;
    float cos_values[N][N]; // Precompute cos values

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cos_values[i][j] = cos((2*i+1)*j*M_PI/(2.0*N));
        }
    }

    // #pragma omp parallel for collapse(2)
    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            // output[u][v] = 0;
            output[u * N + v] = 0;
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    // output[u][v] += input[x][y] * cos_values[x][u] * cos_values[y][v];
                    output[u * N + v] += input[x * N + y] * cos_values[x][u] * cos_values[y][v];
                }
            }
            // output[u][v] *= 2.0 / N;
            output[u * N + v] *= 2.0 / N;
        }
    }

    float temp = 1 / sqrt(2);

    // #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        // output[i][0] *= temp;
        // output[0][i] *= temp;
        output[i * N] *= temp;
        output[i] *= temp;
    }
}