
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10;  // SMALL test first
    double *S = malloc(n * n * sizeof(double));
    double *degree = malloc(n * sizeof(double));
    double *L = malloc(n * n * sizeof(double));

    load_similarity_matrix("data/ans_batch/test1_ddg.txt", S, n, rank);
    compute_degree_matrix(S, degree, n, rank, size);
    laplacian(S, degree, L, n, rank, size);

    if (rank == 0) {
        printf("\nDegree vector:\n");
        for (int i = 0; i < n; i++)
            printf("%f ", degree[i]);
        printf("\n\nLaplacian matrix:\n");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%6.2f ", L[i*n + j]);
            printf("\n");
        }
    }

    free(S); free(degree); free(L);
    MPI_Finalize();
    return 0;
}



