#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>

#include "laplacian.h"
#include "eigensolver.h"
#include "kmeans.h"
#include "metrics.h"

int read_matrix_size(const char *filename) {
    FILE *f = fopen(filename,"r");
    if(!f) { perror("Failed to open"); MPI_Abort(MPI_COMM_WORLD,1); }

    double tmp;
    long long count = 0;
    while (fscanf(f, "%lf%*[, ]", &tmp) == 1) {
        count++;
    }

    fclose(f);
    
    int n = (int)(sqrt((double)count));
    if ((long long)n * n != count) {
        fprintf(stderr, "Input file does not contain a square matrix (values=%lld)", count);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return n;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Defaults
    const char *input_file = "data/ans_batch/test2_ddg.txt";
    int n = -1;          
    int k = 3;
    int clusters = 3;

    // Optional overrides = ./spectral_mpi [file] [k] [clusters]
    if (argc >= 2) input_file = argv[1];
    if (argc >= 3) k = atoi(argv[2]);
    if (argc >= 4) clusters = atoi(argv[3]);

    if (rank == 0) {
        n = read_matrix_size(input_file);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        mkdir("output", 0755);
    }
    MPI_Barrier(MPI_COMM_WORLD); 

    //alloc memory
    double *S = malloc(n * n * sizeof(double));
    double *degree = malloc(n * sizeof(double));
    double *L = malloc(n * n * sizeof(double));
    double *U = malloc(n * k * sizeof(double));
    int *labels = malloc(n * sizeof(int));

    //I add a timing set up - just so we can see each part...
    double t_start, t_total;
    double t_load, t_degree, t_laplacian, t_eigen, t_kmeans;

    MPI_Barrier(MPI_COMM_WORLD);
    t_total = MPI_Wtime();

    //load matirces
    t_start = MPI_Wtime();
    load_similarity_matrix(input_file, S, n, rank);
    t_load = MPI_Wtime() - t_start;

    //compute degree matrix
    MPI_Barrier(MPI_COMM_WORLD); 
    t_start = MPI_Wtime();
    compute_degree_matrix(S, degree, n, rank, size);
    t_degree= MPI_Wtime() - t_start;

    // 2. Laplacian
    MPI_Barrier(MPI_COMM_WORLD); 
    t_start = MPI_Wtime();    
    laplacian(S, degree, L, n, rank, size);
    free(S); // similarity matrix not needed anymore
    S = NULL;
    t_laplacian = MPI_Wtime() - t_start;

    //3.eigenvectors
    // free laplacian on non-root ranks
    if (rank != 0){
        free(L);
        L = NULL;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    compute_eigenvectors(L, U, n, k, rank);
    t_eigen = MPI_Wtime() - t_start;

    // 4. k-means
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    kmeans(U, n, k, clusters, 50, rank, size, labels);
    t_kmeans = MPI_Wtime() - t_start;

    t_total = MPI_Wtime() - t_total;

    double cluster_score = cluster_similarity_score(S, labels, n);

    if(rank==0){
        printf("Cluster quality (intra/inter similarity ratio): %.4f\n", cluster_score);

        // Append to CSV
        FILE *f = fopen("output/performance.csv", "a");
        if(f){
            fprintf(f, "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                    input_file, n, clusters, size,
                    t_load, t_degree, t_laplacian, t_eigen, t_kmeans, t_total,
                    cluster_score);
            fclose(f);
        }
    }

    free(S); free(degree); free(L); free(U); free(labels);
    
    MPI_Finalize();
    return 0;
}
