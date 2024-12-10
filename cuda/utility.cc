#include "utility.h"
#include <cmath>
#include <algorithm>

using namespace std;

float PSNR(Image& origin_image, float* compressed_image){
    int height = origin_image.height;
    int width = origin_image.width;
    int channels = origin_image.channels;

    float max_val = 0;
    float mse = 0;
    float psnr = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int index = y * width * channels + x * channels + c;
                float diff = origin_image.data[y][x][c] - compressed_image[index];
                mse += diff * diff;
                if(max_val < origin_image.data[y][x][c])
                    max_val = origin_image.data[y][x][c];
            }
        }
    }

    mse /= height * width * channels;
    psnr = 10 * log10(max_val * max_val / mse);

    return psnr;
}

float compression_ratio(int origin_size, int compressed_size){
    return origin_size / compressed_size;
}

int* Image_2_pointer(Image& image){
    int height = image.height;
    int width = image.width;
    int channels = image.channels;
    int* img_ptr = new int[height * width * channels];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                img_ptr[y * width * channels + x * channels + c] = image.data[y][x][c];
            }
        }
    }
    
    return img_ptr;
}

Image pointer_2_Image(float* rgb_image, int height, int width, int channels){
    Image rgb_img = {width, height, channels, {}};
    rgb_img.data.resize(height, vector<vector<int>>(width, vector<int>(channels)));

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                rgb_img.data[y][x][c] = static_cast<int>(rgb_image[y * width * channels + x * channels + c]);
            }
        }
    }

    return rgb_img;
}