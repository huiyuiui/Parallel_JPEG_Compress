#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>
#include <assert.h>
#include <time.h>
#include <iostream>
#include "utility.h"
#include "png_io.h"
#include "color_space.h"
#include "quantization.h"
#include "DCT.h"
#include "huffman_code.h"

using namespace std;

//======================
#define DEV_NO 0
cudaDeviceProp prop;

int main(int argc, char* argv[]) {
    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    
    assert(argc == 2);
    const string filename = argv[1];

    // read image
    Image img = read_png(filename);

    // init variable
    int height = img.height;
    int width = img.width;
    int channels = img.channels;
    int full_size = height * width * channels;
    int half_size = height * width + 2 * (height / 2 * width / 2);
    struct timespec start, end;
    double elapsed_time;

    // declare array
    int* host_full_img_i, *host_half_img_i;
    float* host_full_img_f, *host_half_img_f;
    int* dev_full_img_i, *dev_half_img_i;
    float* dev_full_img_f, *dev_half_img_f;
    
    // allocate memory
    host_full_img_i = Image_2_pointer(img);
    host_full_img_f = new float[full_size];
    host_half_img_i = new int[half_size];
    host_half_img_f = new float[half_size];
    cudaMalloc((void**)&dev_full_img_i, full_size * sizeof(int));
    cudaMalloc((void**)&dev_full_img_f, full_size * sizeof(float));
    cudaMalloc((void**)&dev_half_img_i, half_size * sizeof(int));
    cudaMalloc((void**)&dev_half_img_f, half_size * sizeof(float));

    // memory copy from host to device
    cudaMemcpy(dev_full_img_i, host_full_img_i, full_size * sizeof(int), cudaMemcpyHostToDevice);
    init_constant_ycbcr_matrix();
    init_constant_qtable();

    // kernel parameters
    int BLOCK_SIZE = 32;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockNum((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Compression */
    clock_gettime(CLOCK_MONOTONIC, &start);
    // step 1: convert RGB to YCbCr
    RGB_2_YCbCr_kernel<<<blockNum, blockSize>>>(dev_full_img_i, dev_full_img_f, height, width, channels);

    // step 2: chrominance subsample
    chrominance_subsample_kernel<<<blockNum, blockSize>>>(dev_full_img_f, dev_half_img_f, height, width, channels);
    
    // step 3: DCT
    // TODO:
    cudaMemcpy(host_half_img_f, dev_half_img_f, half_size * sizeof(float), cudaMemcpyDeviceToHost);
    // float* dct_image = DCT(host_half_img_f, height, width);
    float* dct_image = DCT_cuda(host_half_img_f, height, width);
    cudaMemcpy(dev_half_img_f, dct_image, half_size * sizeof(float), cudaMemcpyHostToDevice);

    // step 4: quantization
    // dim3 quantizedBlockSize(8, 8);
    // dim3 quantizedBlockNum((width + 8 - 1) / 8, (height + 8 - 1) / 8);
    quantization_kernel<<<blockNum, blockSize>>>(dev_half_img_f, dev_half_img_i, height, width);

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    // memory copy from device back to host
    cudaMemcpy(host_half_img_i, dev_half_img_i, half_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // step 5: huffman code
    auto [encoded_image, codebook] = huffman_encode(host_half_img_i, height * width + 2 * height / 2 * width / 2);

    /* Decompression */
    // step 1: huffman decoding
    int *decoded_image = huffman_decode(encoded_image, codebook , height * width + 2 * height / 2 * width / 2);

    // step 2: dequantization
    int* dequantized_image = dequantization(host_half_img_i, height, width);

    // step 3: IDCT
    float* idct_image = iDCT(dequantized_image, height, width);

    // step 4: chrominance upsample
    float* ycbcr_image = chrominance_upsample(idct_image, height, width, channels);

    // YCbCr to image
    float *rgb_image = YcbCr_2_RGB(ycbcr_image, height, width, channels);

    float psnr = PSNR(img, rgb_image);
    float subsample_compressed_ratio = compression_ratio(full_size, height * width + 2 * height / 2 * width / 2);
    float huffman_compressed_ratio = compression_ratio(full_size * sizeof(int) * 8, encoded_image.length());

    cout << "Compressed PSNR: " << psnr << endl;
    cout << "Compressed ratio after subsample: " << subsample_compressed_ratio << endl;
    cout << "Compressed ratio after huffman encode: " << huffman_compressed_ratio << endl;

    // recover image
    Image rgb_img = pointer_2_Image(rgb_image, height, width, channels);

    write_png("../assets/output.png", rgb_img);

    return 0;
}