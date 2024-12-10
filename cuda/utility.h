// utility.h
#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <string>

using namespace std;

extern int omp_threads;

struct Image {
    int width;
    int height;
    int channels;
    vector<vector<vector<int>>> data; 
};


float PSNR(Image& origin_image, float* compressed_image);

float compression_ratio(int origin_size, int compressed_size);

int* Image_2_pointer(Image& image);

Image pointer_2_Image(float* rgb_image, int height, int width, int channels);

#endif // UTILITY_H
