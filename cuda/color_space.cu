#include "color_space.h"
#include "utility.h"
#include <cuda.h>

using namespace std;

__constant__ float device_YCbCr_matrix[3][3];
__constant__ float device_shift_vector[3];

float* RGB_2_YCbCr(Image& rgb_image){
    int height = rgb_image.height;
    int width = rgb_image.width;
    int channels = rgb_image.channels;
    float* ycbcr_image = new float[height * width * channels];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int index = y * width * channels + x * channels + c;
                int R = rgb_image.data[y][x][0];
                int G = rgb_image.data[y][x][1];
                int B = rgb_image.data[y][x][2];

                ycbcr_image[index] = R * YCbCr_matrix[c][0] +
                                     G * YCbCr_matrix[c][1] +
                                     B * YCbCr_matrix[c][2] +
                                     shift_vector[c];
            }
        }
    }

    return ycbcr_image;
}

float* YcbCr_2_RGB(float* ycbcr_image, int height, int width, int channels){
    float* rgb_image = new float[height * width * channels];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int ycbcr_index = y * width * channels + x * channels;
                int index = y * width * channels + x * channels + c;
                float Y = ycbcr_image[ycbcr_index + 0];
                float Cb = ycbcr_image[ycbcr_index + 1];
                float Cr = ycbcr_image[ycbcr_index + 2];

                rgb_image[index] = (Y - shift_vector[0]) * inv_YCbCr_matrix[c][0] +
                                   (Cb - shift_vector[1]) * inv_YCbCr_matrix[c][1] +
                                   (Cr - shift_vector[2]) * inv_YCbCr_matrix[c][2];
                                   
            }
        }
    }

    return rgb_image;                           
}

float* chrominance_subsample(float* ycbcr_image, int height, int width, int channels){
    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    float* subsampled_image = new float[height * width + 2 * CbCr_height * CbCr_width];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int origin_index = y * width * channels + x * channels;
            int Y_index = y * width + x;
            subsampled_image[Y_index] = ycbcr_image[origin_index];
            
            // subsampling ratio: 4:2:0, so we only need to sample CbCr once every four pixels
            if(y % 2 == 0 && x % 2 == 0){
                int Cb_index = height * width + (y / 2) * CbCr_width + (x / 2);
                int Cr_index = height * width + CbCr_height * CbCr_width + (y / 2) * CbCr_width + (x / 2);
                subsampled_image[Cb_index] = ycbcr_image[origin_index + 1];
                subsampled_image[Cr_index] = ycbcr_image[origin_index + 2];
            }
        }
    }

    return subsampled_image;
}

float* chrominance_upsample(float* subsampled_image, int height, int width, int channels){
    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    float* ycbcr_image = new float[height * width * channels];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int origin_index = y * width * channels + x * channels;
            int Y_index = y * width + x;
            int Cb_index = height * width + (y / 2) * CbCr_width + (x / 2);
            int Cr_index = height * width + CbCr_height * CbCr_width + (y / 2) * CbCr_width + (x / 2);
            ycbcr_image[origin_index + 0] = subsampled_image[Y_index];
            ycbcr_image[origin_index + 1] = subsampled_image[Cb_index];
            ycbcr_image[origin_index + 2] = subsampled_image[Cr_index];
        }
    }
    
    return ycbcr_image;
}


void init_constant_ycbcr_matrix() {
    cudaMemcpyToSymbol(device_YCbCr_matrix, YCbCr_matrix, sizeof(YCbCr_matrix));
    cudaMemcpyToSymbol(device_shift_vector, shift_vector, sizeof(shift_vector));
}

__global__ void RGB_2_YCbCr_kernel(int* rgb_image, float* ycbcr_image, int height, int width, int channels){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float device_YCbCr_matrix[3][3] = {{0.257, 0.504, 0.098},
                                         {-0.148, -0.291, 0.439},
                                         {0.439, -0.368, -0.071}};

    float device_shift_vector[3] = {16.0, 128.0, 128.0};

    int index = y * width * channels + x * channels;
    int R = rgb_image[index + 0];
    int G = rgb_image[index + 1];
    int B = rgb_image[index + 2];

    for (int c = 0; c < channels; c++)
    {
        ycbcr_image[index + c] = R * device_YCbCr_matrix[c][0] + 
                                 G * device_YCbCr_matrix[c][1] + 
                                 B * device_YCbCr_matrix[c][2] + 
                                 device_shift_vector[c];
    }
}

__global__ void chrominance_subsample_kernel(float* ycbcr_image, float* subsampled_image, int height, int width, int channels){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    int origin_index = y * width * channels + x * channels;
    int Y_index = y * width + x;

    subsampled_image[Y_index] = ycbcr_image[origin_index];

    if(y % 2 == 0 && x % 2 == 0){
        int Cb_index = height * width + (y / 2) * CbCr_width + (x / 2);
        int Cr_index = height * width + CbCr_height * CbCr_width + (y / 2) * CbCr_width + (x / 2);
        subsampled_image[Cb_index] = ycbcr_image[origin_index + 1];
        subsampled_image[Cr_index] = ycbcr_image[origin_index + 2];
    }
}
