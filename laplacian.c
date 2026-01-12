#include "laplacian.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_square_matrix(const char *filename, double *S, int n, int rank){
    /* Only rank 0 performs file I/O */
    if (rank == 0){
        FILE *f = fopen(filename, "r");
        if (!f) {
            fprintf(stderr, "Error opening similarity matrix file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j ++) {
                if (fscanf(f, " %lf%*[,]", &S[i*n + j]) != 1) {
                    fprintf (stderr, "Error reading similarity matrix\n");
                    fclose(f);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        fclose(f);    
    }

    /* Broadcast the full matrix to all ranks */
    MPI_Bcast(S, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void compute_degree_matrix (double *S, double *degree, int n, int rank, int size){
    /* Determine row ownership for this rank */
    int rows = n / size;
    int rem = n % size;
    int start = (rank < rem) ? rank * (rows + 1) : rank * rows + rem;
    int end = start + ((rank < rem) ? rows + 1 : rows);

    /* Initialize local degree entries */
    memset(degree, 0, n * sizeof(double));

    /* Compute degree values for owned rows */
    for (int i = start; i < end; i++){
        for (int j = 0; j < n ; j++) {
            degree[i] += S[i * n + j];
        }
    }

    /* Combine partial results across all ranks */
    MPI_Allreduce(MPI_IN_PLACE, degree, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void laplacian (double *S, double *degree, double *L, int n, int rank, int size){
    int rows = n / size;
    int rem = n % size;
    int start = (rank < rem) ? rank * (rows + 1) : rank * rows + rem;
    int end = start + ((rank < rem) ? rows + 1 : rows);

    /* Initialize Laplacian */
    memset(L, 0, n*n * sizeof(double));

    /* Compute local rows of the Laplacian */
    for(int i = start; i < end; i++){
        for(int j = 0; j< n; j++){
            if (i == j) 
                L[i * n + j] = degree[i] - S[i * n + j];
            else 
                L[i * n + j] = -S[i * n + j];
        }
    }

    /* Assemble the full Laplacian on all ranks */
    MPI_Allreduce(MPI_IN_PLACE, L, n * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

