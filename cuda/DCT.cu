#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
using namespace std;

__constant__ float device_temp;

__global__ void DCT_cuda_cal(int N, float *input, float *output, int stride) {
    int offset = blockIdx.x * N * stride + blockIdx.y * N;
    int u = threadIdx.x, v = threadIdx.y;

    __shared__ float cos_values[8][8];
    __shared__ float S_output[8][8];
    __shared__ float S_input[8][8];

    cos_values[u][v] = cos((2 * u + 1) * v * M_PI / (2 * N));
    S_input[u][v] = input[offset + u * stride + v];
    S_output[u][v] = 0;
    __syncthreads();

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            S_output[u][v] += S_input[x][y] * cos_values[x][u] * cos_values[y][v];
        }
    }
    S_output[u][v] *= 2.0 / N;

    if (u == 0) {
        S_output[0][v] *= device_temp;
    }
    if (v == 0) {
        S_output[u][0] *= device_temp;
    }

    output[offset + u * stride + v] = S_output[u][v];
}

void DCT_cuda(float *input, float *output, int height, int width) {
    const int N = 8;
    const int CbCr_height = height / 2, CbCr_width = width / 2;

    float temp = 1 / sqrt(2.0);
    cudaMemcpyToSymbol(device_temp, &temp, sizeof(float));

    dim3 num_threads(N, N);
    dim3 num_blocks(height / N, width / N);
    dim3 num_blocks_CbCr(CbCr_height / N, CbCr_width / N);

    DCT_cuda_cal<<<num_blocks, num_threads>>>(N, input, output, width);
    DCT_cuda_cal<<<num_blocks_CbCr, num_threads>>>(N, input + height * width, output + height * width, CbCr_width);
    DCT_cuda_cal<<<num_blocks_CbCr, num_threads>>>(N, input + height * width + CbCr_height * CbCr_width, output + height * width + CbCr_height * CbCr_width, CbCr_width);

    return;
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
    for (int i = 0; i < height/N; i++) {
        for (int j = 0; j < width/N; j++) {
            DCT_cal(N, input+i*N*width+j*N, output+i*N*width+j*N, width);
        }
    }
    // for Cb and Cr
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