#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <time.h>
#include "png_io.h"
#include "color_space.h"
#include "utility.h"
#include "quantization.h"
#include "huffman_code.h"
#include "DCT.h"

using namespace std;

int omp_threads;

int main(int argc, char** argv) {
    omp_threads = omp_get_max_threads();

    cout << "Number of threads: " << omp_threads << endl;

    assert(argc == 2);
    const string filename = argv[1];

    struct timespec start, end;
    double elapsed_time;

    // read image
    Image img = read_png(filename);

    int height = img.height;
    int width = img.width;
    int channels = img.channels;
    int total_size = height * width * channels;
    
    /* Compression */
    clock_gettime(CLOCK_MONOTONIC, &start);

    // step 1: convert RGB to YCbCr
    // float *ycbcr_image = RGB_2_YCbCr(img);
    float *ycbcr_image = RGB_2_YCbCr_avx512(img);

    // step 2: chrominance subsample
    // float* subsampled_image = chrominance_subsample(ycbcr_image, height, width, channels);
    float* subsampled_image = chrominance_subsample_avx512(ycbcr_image, height, width, channels);
   
    // step 3: DCT
    // float *dct_image = DCT(subsampled_image, height, width);
    float *dct_image = DCT_vec(subsampled_image, height, width);

    // step 4: quantization
    // int* quantized_image = quantization(dct_image, height, width);
    int *quantized_image = quantization_avx512(dct_image, height, width);

    // step 5: huffman encoding
    auto [encoded_image, codebook] = huffman_encode(quantized_image, height * width + 2 * height / 2 * width / 2);

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Compressed time: %f seconds\n", elapsed_time);

    /* Decompression */
    // step 1: huffman decoding
    int *decoded_image = huffman_decode(encoded_image, codebook , height * width + 2 * height / 2 * width / 2);

    // step 2: dequantization
    int* dequantized_image = dequantization(quantized_image, height, width);

    // step 3: IDCT
    float *idct_image = iDCT(dequantized_image, height, width);

    // step 4: chrominance upsample
    ycbcr_image = chrominance_upsample(idct_image, height, width, channels);

    // step 5: convert YCbCr to RGB
    float *rgb_image = YcbCr_2_RGB(ycbcr_image, height, width, channels);

    float psnr = PSNR(img, rgb_image);
    float subsample_compressed_ratio = compression_ratio(total_size, height * width + 2 * height / 2 * width / 2);
    float huffman_compressed_ratio = compression_ratio(total_size * sizeof(int) * 8, encoded_image.length());

    cout << "Compressed PSNR: " << psnr << endl;
    cout << "Compressed ratio after subsample: " << subsample_compressed_ratio << endl;
    cout << "Compressed ratio after huffman encode: " << huffman_compressed_ratio << endl;

    // recover image
    Image rgb_img = {img.width, img.height, img.channels, {}};
    rgb_img.data.resize(height, vector<vector<int>>(width, vector<int>(channels)));

    #pragma omp parallel for num_threads(omp_threads) schedule(static) collapse(3)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int index = y * width * channels + x * channels + c;
                rgb_img.data[y][x][c] = static_cast<int>(rgb_image[index]);
            }
        }
    }

    write_png("../assets/output.png", rgb_img);
}   