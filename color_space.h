// color_space.h
#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

#include "utility.h"

using namespace std;

float* RGB_2_YCbCr(Image& rgb_image);

float* YcbCr_2_RGB(float* ycbcr_image, int height, int width, int channels);

float* chrominance_subsample(float* ycbcr_image, int height, int width, int channels);

float* chrominance_upsample(float* subsampled_image, int height, int width, int channels);


#endif // COLOR_SPACE_H
