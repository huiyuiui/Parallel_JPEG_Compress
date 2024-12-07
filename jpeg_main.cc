#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
#include "png_io.h"

using namespace std;

int main(int argc, char** argv) {
    int mpi_rank, mpi_rank_size, omp_threads;
    MPI_Status stat;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    omp_threads = omp_get_max_threads();

    const string filename = argv[1];

    Image img = read_png(filename);

    // convert to grey image
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int gray = (img.data[y][x][0] + img.data[y][x][1] + img.data[y][x][2]) / 3;
            img.data[y][x][0] = gray;
            img.data[y][x][1] = gray;
            img.data[y][x][2] = gray;
        }
    }

    write_png("output.png", img);
}   