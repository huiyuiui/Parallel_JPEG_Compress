#include "DCT.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstdlib>
#include <immintrin.h>
using namespace std;

void DCT_vec_cal(int N, float *input, float *output, int stride) {
    float cos_values[N][N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cos_values[i][j] = cos((2 * i + 1) * j * M_PI / (2 * N));
        }
    }

    for (int u = 0; u < N; u++) {
        __m256 output_vec = _mm256_setzero_ps();

        for (int x = 0; x < N; x++) {
            __m256 cos_values_xu = _mm256_set1_ps(cos_values[x][u]);

            for (int y = 0; y < N; y++) {
                __m256 cos_values_yv = _mm256_loadu_ps(&cos_values[y][0]);
                __m256 input_vec = _mm256_set1_ps(input[x * stride + y]);

                __m256 temp = _mm256_mul_ps(input_vec, cos_values_xu);
                temp = _mm256_mul_ps(temp, cos_values_yv);
                output_vec = _mm256_add_ps(output_vec, temp);
            }

        }
        output_vec = _mm256_mul_ps(output_vec, _mm256_set1_ps(2.0 / N));
        _mm256_storeu_ps(output + u * stride, output_vec);
    }

    float temp = 1 / sqrt(2.0);
    for (int i = 0; i < N; i++) {
        output[i * stride] *= temp;
        output[i] *= temp;
    }
}

float* DCT_vec(float *input, int height, int width) {
    const int N = 8;
    const int CbCr_height = height / 2, CbCr_width = width / 2;

    float *output = new float[height * width + 2 * CbCr_height * CbCr_width];
    // for Y
    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
    for (int i = 0; i < height/N; i++) {
        for (int j = 0; j < width/N; j++) {
            DCT_vec_cal(N, input+i*N*width+j*N, output+i*N*width+j*N, width);
        }
    }
    // for Cb and Cr
    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
    for (int i = 0; i < CbCr_height/N; i++) {
        for (int j = 0; j < CbCr_width/N; j++) {
            DCT_vec_cal(N, input+height*width+i*N*CbCr_width+j*N, output+height*width+i*N*CbCr_width+j*N, CbCr_width);
            DCT_vec_cal(N, input+height*width+CbCr_height*CbCr_width+i*N*CbCr_width+j*N, output+height*width+CbCr_height*CbCr_width+i*N*CbCr_width+j*N, CbCr_width);
        }
    }

    return output;
}

void DCT_cal(int N, float *input, float *output, int stride) {
    float cos_values[N][N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cos_values[i][j] = cos((2 * i + 1) * j * M_PI / (2 * N));
        }
    }

    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            output[u * stride + v] = 0;
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    output[u * stride + v] += input[x * stride + y] * cos_values[x][u] * cos_values[y][v];
                }
            }
            output[u * stride + v] *= 2.0 / N;
        }
    }

    float temp = 1 / sqrt(2.0);
    for (int i = 0; i < N; i++) {
        output[i * stride] *= temp;
        output[i] *= temp;
    }
}

float* DCT(float *input, int height, int width) {
    const int N = 8;
    const int CbCr_height = height / 2, CbCr_width = width / 2;

    float *output = new float[height * width + 2 * CbCr_height * CbCr_width];
    // for Y
    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
    for (int i = 0; i < height/N; i++) {
        for (int j = 0; j < width/N; j++) {
            DCT_cal(N, input+i*N*width+j*N, output+i*N*width+j*N, width);
        }
    }
    // for Cb and Cr
    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
    for (int i = 0; i < CbCr_height/N; i++) {
        for (int j = 0; j < CbCr_width/N; j++) {
            DCT_cal(N, input+height*width+i*N*CbCr_width+j*N, output+height*width+i*N*CbCr_width+j*N, CbCr_width);
            DCT_cal(N, input+height*width+CbCr_height*CbCr_width+i*N*CbCr_width+j*N, output+height*width+CbCr_height*CbCr_width+i*N*CbCr_width+j*N, CbCr_width);
        }
    }

    return output;
}

void iDCT_cal(int N, int *input, float *output, int stride) {
    float alpha_u, alpha_v, temp = 1 / sqrt(2.0);
    float cos_values[N][N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cos_values[i][j] = cos((2 * i + 1) * j * M_PI / (2 * N));
        }
    }

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            output[x * stride + y] = 0;
            for (int u = 0; u < N; u++) {
                alpha_u = (u == 0) ? temp : 1;
                for (int v = 0; v < N; v++) {
                    alpha_v = (v == 0) ? temp : 1;
                    output[x * stride + y] += alpha_u * alpha_v * input[u * stride + v] * cos_values[x][u] * cos_values[y][v];
                }
            }
            output[x * stride + y] *= 2.0 / N;
        }
    }
}

float* iDCT(int *input, int height, int width) {
    const int N = 8;
    const int CbCr_height = height / 2, CbCr_width = width / 2;

    float *output = new float[height * width + 2 * CbCr_height * CbCr_width];
    // for Y
    for (int i = 0; i < height/N; i++) {
        for (int j = 0; j < width/N; j++) {
            iDCT_cal(N, input+i*N*width+j*N, output+i*N*width+j*N, width);
        }
    }
    // for Cb and Cr
    for (int i = 0; i < CbCr_height/N; i++) {
        for (int j = 0; j < CbCr_width/N; j++) {
            iDCT_cal(N, input+height*width+i*N*CbCr_width+j*N, output+height*width+i*N*CbCr_width+j*N, CbCr_width);
            iDCT_cal(N, input+height*width+CbCr_height*CbCr_width+i*N*CbCr_width+j*N, output+height*width+CbCr_height*CbCr_width+i*N*CbCr_width+j*N, CbCr_width);
        }
    }

    return output;
}