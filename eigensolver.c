#include "eigensolver.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//LAPACK
extern void dsyev_(
    char *jobz, char *uplo,
    int *n, double *a, int *lda,
    double *w,
    double *work, int *lwork,
    int *info
);


void compute_eigenvectors(double *L, double *U, int n, int k, int rank){
    if (rank == 0) {
        /* Copy Laplacian since dsyev overwrites its input */
        double *A  = malloc(n * n * sizeof(double));
        double *w = malloc(n * sizeof(double));
        memcpy(A, L, n * n * sizeof(double));

        /* Workspace query */
        int lwork = -1, info;
        double wkopt;
        char jobz = 'V';
        char uplo = 'U';

        dsyev_(&jobz, &uplo, &n, A, &n, w, &wkopt, &lwork, &info);
        lwork = (int) wkopt;
        double *work = malloc(lwork * sizeof(double));
        
        /* Eigen-decomposition */
        dsyev_(&jobz, &uplo, &n, A, &n, w, work, &lwork, &info);

        if (info != 0){
            fprintf(stderr, "Eigen decomposition failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Extract the first k eigenvectors (smallest eigenvalues) */
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                U[i * k + j] = A[i * n + j];
            
        free(A); 
        free(w); 
        free(work);
    }

    /* Broadcast eigenvectors to all ranks */
    MPI_Bcast(U, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}