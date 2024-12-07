#include "png_io.h"
#include <png.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <assert.h>

using namespace std;

Image read_png(const string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    assert(fp);
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    assert(png);
    png_infop info = png_create_info_struct(png);
    assert(info);
    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    png_read_update_info(png, info);

    vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }
    png_read_image(png, row_pointers.data());
    fclose(fp);

    int channels = png_get_channels(png, info);
    Image image = {width, height, channels, {}};
    image.data.resize(height, vector<vector<int>>(width, vector<int>(channels)));

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                image.data[y][x][c] = static_cast<int>(row[x * channels + c]);
            }
        }
        free(row_pointers[y]);
    }

    png_destroy_read_struct(&png, &info, nullptr);

    cout << "Image loaded: " << width << "x" << height << "x" << channels << endl;

    return image;
}

void write_png(const string& filename, const Image& image) {
    FILE* fp = fopen(filename.c_str(), "wb");
    assert(fp);
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    assert(png);
    png_infop info = png_create_info_struct(png);
    assert(info);
    png_init_io(png, fp);
    png_set_IHDR(
        png, info, image.width, image.height, 8,
        PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, 
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );
    png_set_filter(png, 0, PNG_NO_FILTERS);
    png_write_info(png, info);

    vector<png_bytep> row_pointers(image.height);
    for (int y = 0; y < image.height; y++) {
        row_pointers[y] = (png_bytep)malloc(image.width * image.channels);
        for (int x = 0; x < image.width; x++) {
            for (int c = 0; c < image.channels; c++) {
                row_pointers[y][x * image.channels + c] = static_cast<unsigned char>(image.data[y][x][c]);
            }
        }
    }

    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);

    for (int y = 0; y < image.height; y++) {
        free(row_pointers[y]);
    }
    
    fclose(fp);
    png_destroy_write_struct(&png, &info);
}
