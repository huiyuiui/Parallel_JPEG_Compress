CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -mavx512f
jpeg_main: CC = mpicc
jpeg_main: CXX = mpicxx
jpeg_main: CFLAGS += -fopenmp -pthread

CXXFLAGS = $(CFLAGS)
TARGETS = jpeg_main

.PHONY: all
all: $(TARGETS)

jpeg_main: jpeg_main.o utility.o png_io.o color_space.o quantization.o huffman_code.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

jpeg_main.o: jpeg_main.cc utility.h png_io.h color_space.h quantization.h huffman_code.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

utility.o: utility.cc utility.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

png_io.o: png_io.cc png_io.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

color_space.o: color_space.cc color_space.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

quantization.o: quantization.cc quantization.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

huffman_code.o: huffman_code.cc huffman_code.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o) utility.o png_io.o color_space.o quantization.o huffman_code.o
