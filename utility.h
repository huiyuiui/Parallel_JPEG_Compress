// utility.h
#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <string>

using namespace std;

struct Image {
    int width;
    int height;
    int channels;
    vector<vector<vector<int>>> data; 
};


float PSNR(Image& origin_image, float* compressed_image);

float compression_ratio(int origin_size, int compressed_size);


#endif // UTILITY_H
