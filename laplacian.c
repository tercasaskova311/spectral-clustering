#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>


void load_similarity_matrix(const char *filename, double *S, int n, int rank){
    if (rank == 0){
        FILE *f = fopen(filename, "r");
        if (!f) {
            fprintf(stderr, " Error when loading matrices");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Loading matrices from %s\n", filename);

        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j ++) {
                if (fscanf(f, " %lf%*[,]", &S[i*n + j]) != 1) {
                    fprintf (stderr, "error reading matrix");
                    fclose(f);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }

        fclose(f);
        printf("Loaded similarity matrix of size %d x %d\n", n, n);
    
    }
    //broadcast full matrix to ALL PROCESSES... later each process compute only its assign rows
    MPI_Bcast(S, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void compute_degree_matrix (double *S, double *degree, int n, int rank, int size){
    int rows_per_proc = n / size;
    int start_row;
    int end_row;

    int remainder = n % size;
    if (rank < remainder) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc + 1;
    } else {
        start_row = rank * rows_per_proc + remainder;
        end_row = start_row + rows_per_proc;
    }

    //degree vector to zero....
    memset(degree, 0, n * sizeof(double));

    //each process computes ITS ROW
    for (int i = start_row; i < end_row; i++){
        double row_sum = 0.0;
        for (int j = 0; j < n ; j++) {
            row_sum += S[i * n + j];
        }
        degree[i] = row_sum;
    }

    MPI_Allreduce(MPI_IN_PLACE, degree, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //now I have everything back - creating degree vector from all sums...
}


void laplacian (double *S, double *degree, double *L, int n, int rank, int size){
    int rows_per_proc = n / size;
    int start_row;
    int end_row;

    int remainder = n % size;  
    if (rank < remainder) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc + 1;
    } else {
        start_row = rank * rows_per_proc + remainder;
        end_row = start_row + rows_per_proc;
    }

    memset(L, 0, n*n * sizeof(double));

    for(int i = start_row; i < end_row; i++){
        for(int j = 0; j< n; j++){
            if (i == j) {
                L[i * n + j] = degree[i] - S[i * n + j];
            } else {
                L[i * n + j] = -S[i * n + j];
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, L, n * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

