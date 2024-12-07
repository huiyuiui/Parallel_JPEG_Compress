#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include "png_io.h"
#include "color_space.h"
#include "utility.h"
#include "quantization.h"
#include "huffman_code.h"

using namespace std;

int main(int argc, char** argv) {
    int mpi_rank, mpi_rank_size, omp_threads;
    MPI_Status stat;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    omp_threads = omp_get_max_threads();

    cout << "Number of processes: " << mpi_rank_size << endl;
    cout << "Number of threads: " << omp_threads << endl;

    assert(argc == 2);
    const string filename = argv[1];

    // read image
    Image img = read_png(filename);

    int height = img.height;
    int width = img.width;
    int channels = img.channels;
    int total_size = height * width * channels;

    /* Compression*/
    // step 1: convert RGB to YCbCr
    float *ycbcr_image = RGB_2_YCbCr(img);

    // step 2: chrominance subsample
    float* subsampled_image = chrominance_subsample(ycbcr_image, height, width, channels);

    // step 3: DCT
    // TODO:

    // step 4: quantization
    int* quantized_image = quantization(subsampled_image, height, width);

    // step 5: huffman encoding
    auto [encoded_image, codebook] = huffman_encode(quantized_image, height * width + 2 * height / 2 * width / 2);

    /* Decompression */
    // step 1: huffman decoding
    int *decoded_image = huffman_decode(encoded_image, codebook);

    // step 2: dequantization
    int* dequantized_image = dequantization(decoded_image, height, width);

    // step 3: IDCT
    // TODO:

    // step 4: chrominance upsample
    ycbcr_image = chrominance_upsample(subsampled_image, height, width, channels);

    // step 5: convert YCbCr to RGB
    float *rgb_image = YcbCr_2_RGB(ycbcr_image, height, width, channels);

    float psnr = PSNR(img, rgb_image);
    float subsample_compressed_ratio = compression_ratio(total_size, height * width + 2 * height / 2 * width / 2);
    float huffman_compressed_ration = compression_ratio(total_size, encoded_image.length());

    cout << "Compressed PSNR: " << psnr << endl;
    cout << "Compressed ratio after subsample: " << subsample_compressed_ratio << endl;
    cout << "Compressed ratio after huffman encode: " << huffman_compressed_ration << endl;

    // recover image
    Image rgb_img = {img.width, img.height, img.channels, {}};
    rgb_img.data.resize(height, vector<vector<int>>(width, vector<int>(channels)));

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

    write_png("./assets/output.png", rgb_img);
}   