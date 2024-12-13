#include "quantization.h"
#include <math.h>
#include <immintrin.h>
#include <iostream>
#include "utility.h"

using namespace std;

int* quantization(float* dct_image, int height, int width){
    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    int* quantized_image = new int[height * width + 2 * CbCr_height * CbCr_width];

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
    for (int y = 0; y < height; y+=8)
    {
        for (int x = 0; x < width; x+=8)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    int Y_index = (y + i) * width + (x + j);
                    quantized_image[Y_index] = static_cast<int>(round(dct_image[Y_index] / (Luminance_Qtable[i][j] * scale_factor)));

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

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
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


int* quantization_avx512(float *dct_image, int height, int width){
    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    int Cb_base_index = height * width;
    int Cr_base_index = height * width + CbCr_height * CbCr_width;
    int *quantized_image = new int[height * width + 2 * CbCr_height * CbCr_width];
    // Vectorization: 4 * 16 = 8 * 8 = 64
    __m512 Lumi_Qtable[4], Chromi_Qtable[4];
    __m512 scale_vec = _mm512_set1_ps(scale_factor);

    // init quantization table
    for (int i = 0; i < 4; i++) {
        Lumi_Qtable[i] = _mm512_mul_ps(scale_vec, _mm512_cvtepi32_ps(_mm512_set_epi32(
            Luminance_Qtable[i * 2 + 1][7], Luminance_Qtable[i * 2 + 1][6], Luminance_Qtable[i * 2 + 1][5], Luminance_Qtable[i * 2 + 1][4],
            Luminance_Qtable[i * 2 + 1][3], Luminance_Qtable[i * 2 + 1][2], Luminance_Qtable[i * 2 + 1][1], Luminance_Qtable[i * 2 + 1][0],
            Luminance_Qtable[i * 2][7], Luminance_Qtable[i * 2][6], Luminance_Qtable[i * 2][5], Luminance_Qtable[i * 2][4],
            Luminance_Qtable[i * 2][3], Luminance_Qtable[i * 2][2], Luminance_Qtable[i * 2][1], Luminance_Qtable[i * 2][0])));
        Chromi_Qtable[i] = _mm512_cvtepi32_ps(_mm512_set_epi32(
            Chrominance_Qtable[i * 2 + 1][7], Chrominance_Qtable[i * 2 + 1][6], Chrominance_Qtable[i * 2 + 1][5], Chrominance_Qtable[i * 2 + 1][4],
            Chrominance_Qtable[i * 2 + 1][3], Chrominance_Qtable[i * 2 + 1][2], Chrominance_Qtable[i * 2 + 1][1], Chrominance_Qtable[i * 2 + 1][0],
            Chrominance_Qtable[i * 2][7], Chrominance_Qtable[i * 2][6], Chrominance_Qtable[i * 2][5], Chrominance_Qtable[i * 2][4],
            Chrominance_Qtable[i * 2][3], Chrominance_Qtable[i * 2][2], Chrominance_Qtable[i * 2][1], Chrominance_Qtable[i * 2][0]));
    }

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
    for (int y = 0; y < height; y+=8)
    {
        for (int x = 0; x < width; x+=8)
        {
            for (int i = 0; i < 4; i++)
            {
                int first_row_index = (y + i * 2) * width + x;
                int second_row_index = (y + i * 2 + 1) * width + x;
                // init Y
                __m512 Y = _mm512_set_ps(
                    dct_image[second_row_index + 7], dct_image[second_row_index + 6], dct_image[second_row_index + 5], dct_image[second_row_index + 4],
                    dct_image[second_row_index + 3], dct_image[second_row_index + 2], dct_image[second_row_index + 1], dct_image[second_row_index + 0],
                    dct_image[first_row_index + 7], dct_image[first_row_index + 6], dct_image[first_row_index + 5], dct_image[first_row_index + 4],
                    dct_image[first_row_index + 3], dct_image[first_row_index + 2], dct_image[first_row_index + 1], dct_image[first_row_index + 0]);
                // quantize Y
                __m512 Q_Y = _mm512_div_ps(Y, Lumi_Qtable[i]);
                __m512i Q_Y_int = _mm512_cvtps_epi32(_mm512_roundscale_ps(Q_Y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                // store Y
                int store_arr[16];
                _mm512_storeu_epi32((__m512i *)store_arr, Q_Y_int);
                for (int j = 0; j < 8; j++)
                {
                    quantized_image[first_row_index + j] = store_arr[j];
                    quantized_image[second_row_index + j] = store_arr[8 + j];
                }

                if(y % 16 == 0 && y / 2 < CbCr_height && x % 16 == 0 && x / 2 < CbCr_width){ // for better load balancing
                    int Cb_first_row_index = Cb_base_index + (y / 2 + i * 2) * CbCr_width + x / 2;
                    int Cb_second_row_index = Cb_base_index + (y / 2 + i * 2 + 1) * CbCr_width + x / 2;
                    int Cr_first_row_index = Cr_base_index + (y / 2 + i * 2) * CbCr_width + x / 2;
                    int Cr_second_row_index = Cr_base_index + (y / 2 + i * 2 + 1) * CbCr_width + x / 2;
                    // init Cb, Cr
                    __m512 Cb = _mm512_set_ps(
                        dct_image[Cb_second_row_index + 7], dct_image[Cb_second_row_index + 6], dct_image[Cb_second_row_index + 5], dct_image[Cb_second_row_index + 4],
                        dct_image[Cb_second_row_index + 3], dct_image[Cb_second_row_index + 2], dct_image[Cb_second_row_index + 1], dct_image[Cb_second_row_index + 0],
                        dct_image[Cb_first_row_index + 7], dct_image[Cb_first_row_index + 6], dct_image[Cb_first_row_index + 5], dct_image[Cb_first_row_index + 4],
                        dct_image[Cb_first_row_index + 3], dct_image[Cb_first_row_index + 2], dct_image[Cb_first_row_index + 1], dct_image[Cb_first_row_index + 0]);
                    __m512 Cr = _mm512_set_ps(
                        dct_image[Cr_second_row_index + 7], dct_image[Cr_second_row_index + 6], dct_image[Cr_second_row_index + 5], dct_image[Cr_second_row_index + 4],
                        dct_image[Cr_second_row_index + 3], dct_image[Cr_second_row_index + 2], dct_image[Cr_second_row_index + 1], dct_image[Cr_second_row_index + 0],
                        dct_image[Cr_first_row_index + 7], dct_image[Cr_first_row_index + 6], dct_image[Cr_first_row_index + 5], dct_image[Cr_first_row_index + 4],
                        dct_image[Cr_first_row_index + 3], dct_image[Cr_first_row_index + 2], dct_image[Cr_first_row_index + 1], dct_image[Cr_first_row_index + 0]);
                    // quantize Cb, Cr
                    __m512 Q_Cb = _mm512_div_ps(Cb, Chromi_Qtable[i]);
                    __m512 Q_Cr = _mm512_div_ps(Cr, Chromi_Qtable[i]);
                    __m512i Q_Cb_int = _mm512_cvtps_epi32(_mm512_roundscale_ps(Q_Cb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    __m512i Q_Cr_int = _mm512_cvtps_epi32(_mm512_roundscale_ps(Q_Cr, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    // store Cb, Cr
                    int store_arr_Cb[16];
                    int store_arr_Cr[16];
                    _mm512_storeu_epi32((__m512i *)store_arr_Cb, Q_Cb_int);
                    _mm512_storeu_epi32((__m512i *)store_arr_Cr, Q_Cr_int);
                    for (int j = 0; j < 8; j++)
                    {
                        quantized_image[Cb_first_row_index + j] = store_arr_Cb[j];
                        quantized_image[Cb_second_row_index + j] = store_arr_Cb[8 + j];
                        quantized_image[Cr_first_row_index + j] = store_arr_Cr[j];
                        quantized_image[Cr_second_row_index + j] = store_arr_Cr[8 + j];
                    }
                }
            }
        }
    }

    return quantized_image;
}