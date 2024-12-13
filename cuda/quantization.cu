#include "quantization.h"
#include <math.h>
#include <immintrin.h>
#include <iostream>
#include "utility.h"

using namespace std;

__constant__ int device_lumi_qtable[8][8];
__constant__ int device_chromi_qtable[8][8];
__constant__ float device_scale_factor;

int* quantization(float* dct_image, int height, int width){
    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    int* quantized_image = new int[height * width + 2 * CbCr_height * CbCr_width];

    for (int y = 0; y < height; y+=8)
    {
        for (int x = 0; x < width; x+=8)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    int Y_index = (y + i) * width + (x + j);
                    quantized_image[Y_index] = static_cast<int>(round(dct_image[Y_index] / Luminance_Qtable[i][j]));

                    if(y < CbCr_height && x < CbCr_width){
                        int Cb_index = height * width + (y + i) * CbCr_width + (x + j);
                        int Cr_index = height * width + CbCr_height * CbCr_width + (y + i) * CbCr_width + (x + j);
                        quantized_image[Cb_index] = static_cast<int>(round(dct_image[Cb_index] / Chrominance_Qtable[i][j]));
                        quantized_image[Cr_index] = static_cast<int>(round(dct_image[Cr_index] / Chrominance_Qtable[i][j]));
                    }
                }
            }
        }
    }

    return quantized_image;
}

int* dequantization(int* idct_image, int height, int width){
    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    int* dequantized_image = new int[height * width + 2 * CbCr_height * CbCr_width];

    for (int y = 0; y < height; y+=8)
    {
        for (int x = 0; x < width; x+=8)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    int Y_index = (y + i) * width + (x + j);
                    dequantized_image[Y_index] = idct_image[Y_index] * (Luminance_Qtable[i][j] * scale_factor);

                    if(y < CbCr_height && x < CbCr_width){
                        int Cb_index = height * width + (y + i) * CbCr_width + (x + j);
                        int Cr_index = height * width + CbCr_height * CbCr_width + (y + i) * CbCr_width + (x + j);
                        dequantized_image[Cb_index] = idct_image[Cb_index] * Chrominance_Qtable[i][j];
                        dequantized_image[Cr_index] = idct_image[Cr_index] * Chrominance_Qtable[i][j];
                    }
                }
            }
        }
    }

    return dequantized_image;
}


void init_constant_qtable(){
    cudaMemcpyToSymbol(device_lumi_qtable, Luminance_Qtable, sizeof(Luminance_Qtable));
    cudaMemcpyToSymbol(device_chromi_qtable, Chrominance_Qtable, sizeof(Chrominance_Qtable));
    cudaMemcpyToSymbol(device_scale_factor, &scale_factor, sizeof(scale_factor));
}

/* BlockSize(8, 8) version*/
// __global__ void quantization_kernel(float* dct_image, int* quantized_image, int height, int width){
//     int i = threadIdx.y;
//     int j = threadIdx.x;
//     int x = blockIdx.x * blockDim.x + j;
//     int y = blockIdx.y * blockDim.y + i;

//     int CbCr_height = height / 2;
//     int CbCr_width = width / 2;
//     int Y_index = y * width + x;
//     quantized_image[Y_index] = static_cast<int>(roundf(dct_image[Y_index] / device_lumi_qtable[i][j]));

//     if(y < CbCr_height && x < CbCr_width){
//         int Cb_index = height * width + y * CbCr_width + x;
//         int Cr_index = height * width + CbCr_height * CbCr_width + y * CbCr_width + x;
//         quantized_image[Cb_index] = static_cast<int>(roundf(dct_image[Cb_index] / device_chromi_qtable[i][j]));
//         quantized_image[Cr_index] = static_cast<int>(roundf(dct_image[Cr_index] / device_chromi_qtable[i][j]));
//     }
// }

/* BlockSize(32, 32) version*/
__global__ void quantization_kernel(float* dct_image, int* quantized_image, int height, int width){
    int i = threadIdx.y;
    int j = threadIdx.x;
    int x = blockIdx.x * blockDim.x + j;
    int y = blockIdx.y * blockDim.y + i;

    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    int Y_index = y * width + x;
    quantized_image[Y_index] = static_cast<int>(roundf(dct_image[Y_index] / (device_lumi_qtable[i % 8][j % 8] * device_scale_factor)));

    if(y < CbCr_height && x < CbCr_width){
        int Cb_index = height * width + y * CbCr_width + x;
        int Cr_index = height * width + CbCr_height * CbCr_width + y * CbCr_width + x;
        quantized_image[Cb_index] = static_cast<int>(roundf(dct_image[Cb_index] / device_chromi_qtable[i % 8][j % 8]));
        quantized_image[Cr_index] = static_cast<int>(roundf(dct_image[Cr_index] / device_chromi_qtable[i % 8][j % 8]));
    }
}