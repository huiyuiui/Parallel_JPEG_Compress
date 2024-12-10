// color_space.h
#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

#include "utility.h"
#include <cuda.h>

using namespace std;

inline float YCbCr_matrix[3][3] = {{0.257, 0.504, 0.098},
                                  {-0.148, -0.291, 0.439},
                                  {0.439, -0.368, -0.071}};

inline float inv_YCbCr_matrix[3][3] = {{1.164, 0, 1.596},
                                    {1.164, -0.392, -0.813},
                                    {1.164, 2.017, 0}};

inline float shift_vector[3] = {16.0, 128.0, 128.0};

float* RGB_2_YCbCr(Image& rgb_image);

float* YcbCr_2_RGB(float* ycbcr_image, int height, int width, int channels);

float* chrominance_subsample(float* ycbcr_image, int height, int width, int channels);

float* chrominance_upsample(float* subsampled_image, int height, int width, int channels);

void init_constant_ycbcr_matrix();

__global__ void RGB_2_YCbCr_kernel(int* rgb_image, float* ycbcr_image,int height, int width, int channels);

__global__ void chrominance_subsample_kernel(float* ycbcr_image, float* subsampled_image, int height, int width, int channels);

#endif // COLOR_SPACE_H
