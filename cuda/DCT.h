#ifndef _DCT_H_
#define _DCT_H_

#include "utility.h"

void DCT_cuda(float *input, float *output, int height, int width);

float* DCT(float *input, int height, int width);
float* iDCT(int *input, int height, int width);

#endif