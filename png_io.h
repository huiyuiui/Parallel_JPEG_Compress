// png_io.h
#ifndef PNG_IO_H
#define PNG_IO_H

#include <vector>
#include <string>

using namespace std;

struct Image {
    int width;
    int height;
    int channels;
    vector<vector<vector<int>>> data; 
};

Image read_png(const string& filename);

void write_png(const string& filename, const Image& image);

#endif // PNG_IO_H
