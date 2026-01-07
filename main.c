#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Can be changed to use our data folder
void generate_data(double *X, int n, int d, int rank) {
    for (int i = 0; i < n * d; i++) {
        X[i] = drand48() + rank;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Trial input
    int N = 1000;   // total points
    int d = 2;      // dimensions
    int k = 2;      // clusters

    int n_local = N / size;
    double *X_local = malloc(n_local * d * sizeof(double));

    if (!X_local) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    generate_data(X_local, n_local, d, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("MPI program running with %d ranks\n", size);
    }

    // TODO:
    // 1. Similarity matrix
    // 2. Laplacian
    // 3. Eigenvectors
    // 4. k-means

    free(X_local);
    MPI_Finalize();
    return 0;
}