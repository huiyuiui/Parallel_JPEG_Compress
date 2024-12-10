// png_io.h
#ifndef PNG_IO_H
#define PNG_IO_H

#include <string>
#include "utility.h"

using namespace std;

Image read_png(const string& filename);

void write_png(const string& filename, const Image& image);

#endif // PNG_IO_H
