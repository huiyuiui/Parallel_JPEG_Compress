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

jpeg_main: jpeg_main.o png_io.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

jpeg_main.o: jpeg_main.cc png_io.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

png_io.o: png_io.cc png_io.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o) png_io.o
