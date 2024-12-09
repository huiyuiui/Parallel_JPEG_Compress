#include "color_space.h"
#include "utility.h"
#include <immintrin.h>

using namespace std;

float* RGB_2_YCbCr(Image& rgb_image){
    int height = rgb_image.height;
    int width = rgb_image.width;
    int channels = rgb_image.channels;
    float* ycbcr_image = new float[height * width * channels];

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(3)
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

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(3)
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

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
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

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(2)
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


float *RGB_2_YCbCr_avx512(Image &rgb_image){
    int height = rgb_image.height;
    int width = rgb_image.width;
    int channels = rgb_image.channels;
    float* ycbcr_image = new float[height * width * channels];
    int vec_size = 16; // avx512: 16 int or float per register

    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x += vec_size) 
        {
            __m512 R, G, B;
            __m512 Y, Cb, Cr;

            // init R, G, B
            R = _mm512_cvtepi32_ps(_mm512_set_epi32(
                rgb_image.data[y][x + 15][0], rgb_image.data[y][x + 14][0], rgb_image.data[y][x + 13][0], rgb_image.data[y][x + 12][0],
                rgb_image.data[y][x + 11][0], rgb_image.data[y][x + 10][0], rgb_image.data[y][x + 9][0], rgb_image.data[y][x + 8][0],
                rgb_image.data[y][x + 7][0], rgb_image.data[y][x + 6][0], rgb_image.data[y][x + 5][0], rgb_image.data[y][x + 4][0],
                rgb_image.data[y][x + 3][0], rgb_image.data[y][x + 2][0], rgb_image.data[y][x + 1][0], rgb_image.data[y][x][0]));

            G = _mm512_cvtepi32_ps(_mm512_set_epi32(
                rgb_image.data[y][x + 15][1], rgb_image.data[y][x + 14][1], rgb_image.data[y][x + 13][1], rgb_image.data[y][x + 12][1],
                rgb_image.data[y][x + 11][1], rgb_image.data[y][x + 10][1], rgb_image.data[y][x + 9][1], rgb_image.data[y][x + 8][1],
                rgb_image.data[y][x + 7][1], rgb_image.data[y][x + 6][1], rgb_image.data[y][x + 5][1], rgb_image.data[y][x + 4][1],
                rgb_image.data[y][x + 3][1], rgb_image.data[y][x + 2][1], rgb_image.data[y][x + 1][1], rgb_image.data[y][x][1]));

            B = _mm512_cvtepi32_ps(_mm512_set_epi32(
                rgb_image.data[y][x + 15][2], rgb_image.data[y][x + 14][2], rgb_image.data[y][x + 13][2], rgb_image.data[y][x + 12][2],
                rgb_image.data[y][x + 11][2], rgb_image.data[y][x + 10][2], rgb_image.data[y][x + 9][2], rgb_image.data[y][x + 8][2],
                rgb_image.data[y][x + 7][2], rgb_image.data[y][x + 6][2], rgb_image.data[y][x + 5][2], rgb_image.data[y][x + 4][2],
                rgb_image.data[y][x + 3][2], rgb_image.data[y][x + 2][2], rgb_image.data[y][x + 1][2], rgb_image.data[y][x][2]));

            // Calculate Y
            Y = _mm512_mul_ps(R, _mm512_set1_ps(YCbCr_matrix[0][0])); // R * YCbCr_matrix[0][0]
            Y = _mm512_fmadd_ps(G, _mm512_set1_ps(YCbCr_matrix[0][1]), Y); // + G * YCbCr_matrix[0][1]
            Y = _mm512_fmadd_ps(B, _mm512_set1_ps(YCbCr_matrix[0][2]), Y); // + B * YCbCr_matrix[0][2]
            Y = _mm512_add_ps(Y, _mm512_set1_ps(shift_vector[0])); // + shift_vector[0]

            // Calculate Cb
            Cb = _mm512_mul_ps(R, _mm512_set1_ps(YCbCr_matrix[1][0])); // R * YCbCr_matrix[1][0]
            Cb = _mm512_fmadd_ps(G, _mm512_set1_ps(YCbCr_matrix[1][1]), Cb); // + G * YCbCr_matrix[1][0]
            Cb = _mm512_fmadd_ps(B, _mm512_set1_ps(YCbCr_matrix[1][2]), Cb); // + B * YCbCr_matrix[1][0]
            Cb = _mm512_add_ps(Cb, _mm512_set1_ps(shift_vector[1])); // + shift_vector[1]

            // Calculate Cr
            Cr = _mm512_mul_ps(R, _mm512_set1_ps(YCbCr_matrix[2][0])); // R * YCbCr_matrix[2][0]
            Cr = _mm512_fmadd_ps(G, _mm512_set1_ps(YCbCr_matrix[2][1]), Cr); // + G * YCbCr_matrix[2][0]
            Cr = _mm512_fmadd_ps(B, _mm512_set1_ps(YCbCr_matrix[2][2]), Cr); // + B * YCbCr_matrix[2][0]
            Cr = _mm512_add_ps(Cr, _mm512_set1_ps(shift_vector[2])); // + shift_vector[2]

            // store value as Y Cb Cr
            float* Y_ptr = &ycbcr_image[y * width + x];
            float* Cb_ptr = &ycbcr_image[height * width + y * width + x];
            float *Cr_ptr = &ycbcr_image[2 * height * width + y * width + x];
            _mm512_storeu_ps(Y_ptr, Y);  // store Y
            _mm512_storeu_ps(Cb_ptr, Cb); // store Cb
            _mm512_storeu_ps(Cr_ptr, Cr); // store Cr
        }
    }

    return ycbcr_image;
}


float *chrominance_subsample_avx512(float *ycbcr_image, int height, int width, int channels){
    int CbCr_height = height / 2;
    int CbCr_width = width / 2;
    float* subsampled_image = new float[height * width + 2 * CbCr_height * CbCr_width];
    int vec_size = 16; // avx512: 16 int or float per register
    int Cb_base = height * width;
    int Cr_base = 2 * height * width;

    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x += vec_size)
        {   
            // direct copy Y
            __m512 Y = _mm512_loadu_ps(&ycbcr_image[y * width + x]);
            _mm512_storeu_ps(&subsampled_image[y * width + x], Y);

            // subsampling ratio: 4:2:0, so we only need to sample CbCr once every four pixels
            if(y % 2 == 0){
                int index = y * width;
                // only avx256 is enough to copy CbCr
                __m256 Cb, Cr;
                Cb = _mm256_set_ps(
                    ycbcr_image[Cb_base + index + (x + 14)], ycbcr_image[Cb_base + index + (x + 12)],
                    ycbcr_image[Cb_base + index + (x + 10)], ycbcr_image[Cb_base + index + (x + 8)],
                    ycbcr_image[Cb_base + index + (x + 6)], ycbcr_image[Cb_base + index + (x + 4)],
                    ycbcr_image[Cb_base + index + (x + 2)], ycbcr_image[Cb_base + index + x]);
                Cr = _mm256_set_ps(
                    ycbcr_image[Cr_base + index + (x + 14)], ycbcr_image[Cr_base + index + (x + 12)],
                    ycbcr_image[Cr_base + index + (x + 10)], ycbcr_image[Cr_base + index + (x + 8)],
                    ycbcr_image[Cr_base + index + (x + 6)], ycbcr_image[Cr_base + index + (x + 4)],
                    ycbcr_image[Cr_base + index + (x + 2)], ycbcr_image[Cr_base + index + x]);

                // store CbCr
                int Cb_index = height * width + (y / 2) * CbCr_width + (x / 2);
                int Cr_index = height * width + CbCr_height * CbCr_width + (y / 2) * CbCr_width + (x / 2);
                _mm256_storeu_ps(&subsampled_image[Cb_index], Cb);
                _mm256_storeu_ps(&subsampled_image[Cr_index], Cr);
            }
        }
    }

    return subsampled_image;
}
