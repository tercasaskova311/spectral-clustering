//LAPACK 
extern void dsyev_(
    char *jobz, char *uplo,
    int *n, double *a, int *lda,
    double *w,
    double *work, int *lwork,
    int *info
);


void compute_eigenvectors(double *L, double *eigenvecs, int n, int k, int rank){
    double *A = NULL;
    double *eigvals = NULL;

    if (rank == 0) {
        A = malloc(n * n * sizeof(double));
        eigvals = malloc(n * sizeof(double));
        memcpy(A, L, n * n * sizeof(double));
    }

    int lwork = -1;
    double wkopt;
    int info;
    char jobz = 'V';
    char uplo = 'U';

    dsyev_(&jobz, &uplo, &n, A, &n, eigvals, &wkopt, &lwork, &info);

    lwork = (int) wkopt;
    double *work = malloc(lwork * sizeof(double));

    dsyev_(&jobz, &uplo, &n, A, &n, eigvals, work, &lwork, &info);

    if (info != 0){
        fprintf(stderr, "Eigen decomposition failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // the first k eigenvectors which are smaller
    for (int i = 0; i < n; i++){
        for (int j = 0; j < k; j++){
            eigvecs[i * k + j] = A[i * n + j];
        }
    }

    free(work);
    free(A);
    free(eigvals);

    // broadcast eigenvectors to all ranks
    MPI_Bcast(eigvecs, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}