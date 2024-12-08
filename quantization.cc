#include "quantization.h"
#include <math.h>

using namespace std;

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
                    dequantized_image[Y_index] = idct_image[Y_index] * Luminance_Qtable[i][j];

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