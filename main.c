#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "laplacian.h"
#include "eigensolver.h"
#include "kmeans.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10;  // number of points
    int k = 2; // number of eigenvectors
    int clusters= 2; // number of clusters

    double *S = malloc(n * n * sizeof(double));
    double *degree = malloc(n * sizeof(double));
    double *L = malloc(n * n * sizeof(double));
    double *U = malloc(n * k * sizeof(double));
    int *labels = malloc(n * sizeof(int));

    // 1. Similarity matrix
    load_similarity_matrix("data/ans_batch/test1_ddg.txt", S, n, rank);
    compute_degree_matrix(S, degree, n, rank, size);

    // 2. Laplacian
    laplacian(S, degree, L, n, rank, size);

    // 3. Eigenvectors
    compute_eigenvectors(L, U, n, k, rank);

    // 4. k-means
    kmeans(U, n, k, clusters, 50, rank, size, labels);

    /*if (rank == 0) {
        printf("\nDegree vector:\n");
        for (int i = 0; i < n; i++)
            printf("%f ", degree[i]);
        printf("\n\nLaplacian matrix:\n");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%6.2f ", L[i*n + j]);
            printf("\n");
        }
    }    */
    if (rank==0){
        printf("Spectral clustering completed. \n");
    }

    free(S); free(degree); free(L); free(U); free(labels);
    MPI_Finalize();
    return 0;
}