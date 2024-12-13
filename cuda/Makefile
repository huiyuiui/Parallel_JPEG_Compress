NVFLAGS = -std=c++17 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler="-fopenmp"
HIPCCFLAGS = -std=c++17 -O3 --offload-arch=gfx90a

LDFLAGS = -lm -lpng
EXES = cuda_main

# Source files
SRC = cuda_main.cu utility.cc png_io.cc color_space.cu quantization.cu DCT.cu huffman_code.cu

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

$(EXES): $(SRC)
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $^
