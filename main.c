#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>

#include "laplacian.h"
#include "eigensolver.h"
#include "kmeans.h"
#include "metrics.h"
#include "read_matrix_size.h"
#include "compute_similarity.h"

#ifndef ENABLE_METRICS
#define ENABLE_METRICS 1
#endif

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Defaults
    const char *input_file = "data/ans_batch/test2_wam.txt";
    int n = -1;          
    int k = 3;
    int clusters = 3;
    int cols = -1;
    int is_feature = 0;
    double sigma = -1.0;  // Negative means auto-compute

    // Optional overrides = ./spectral_mpi [file] [k] [clusters] [sigma]
    if (argc >= 2) input_file = argv[1];
    if (argc >= 3) k = atoi(argv[2]);
    if (argc >= 4) clusters = atoi(argv[3]);
    if (argc >= 5) sigma = atof(argv[4]);  // Optional sigma parameter

    if (rank == 0) {
        n = get_square_matrix_size(input_file);
        // not square matrix => get the size...
        if (n == -1) {
            n = get_feature_matrix_size(input_file, &cols);
            if (n<= 0 || cols <= 0) {
                fprintf(stderr, "cannot read matrix\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            is_feature = 1;
        }
        if (k > n || clusters > n) {
            fprintf(stderr, "Error: k or clusters larger than n\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&is_feature, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);           
    MPI_Bcast(&clusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) mkdir("output", 0755);
    MPI_Barrier(MPI_COMM_WORLD); 

    //alloc memory
    double *S = malloc(n * n * sizeof(double));
    double *degree = malloc(n * sizeof(double));
    double *L = malloc(n * n * sizeof(double));
    double *U = malloc(n * k * sizeof(double));
    int *labels = malloc(n * sizeof(int));

    if (!S || !degree || !L || !U || !labels) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //I add a timing set up - just so we can see each part...
    double t_start, t_total;
    double t_load, t_degree, t_laplacian, t_eigen, t_kmeans;

    MPI_Barrier(MPI_COMM_WORLD);
    t_total = MPI_Wtime();

    //load matirces
    t_start = MPI_Wtime();
    //Load / compute similarity - some matrices in the data are squared...
    if (is_feature) {
        if (rank == 0) {
            double *X = malloc(n * cols * sizeof(double));
            if (!X) MPI_Abort(MPI_COMM_WORLD, 1);

            load_feature_matrix(input_file, X, n, cols);
            compute_similarity_matrix(X, S, n, cols, sigma);
            free(X);
        }
        MPI_Bcast(S, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        load_square_matrix(input_file, S, n, rank);  // ALL RANKS call this
    }    
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

    // metrics ======================================
    if(rank == 0){
        printf("\n=== Spectral Clustering Results ===\n");
        printf("Dataset: %s\n", input_file);
        if (is_feature) {
            printf("Input: %d x %d feature matrix\n", n, cols);
            printf("Computed similarity matrix: %d x %d\n", n, n);
            printf("Sigma (RBF bandwidth): %.4f\n", sigma);
        } else {
            printf("Matrix size: %d x %d\n", n, n);
        }
        printf("Eigenvectors (k): %d\n", k);
        printf("Clusters: %d\n", clusters);
        printf("MPI processes: %d\n", size);
        printf("\n--- Timing ---\n");
        printf("Load matrix:    %.6f s\n", t_load);
        printf("Degree matrix:  %.6f s\n", t_degree);
        printf("Laplacian:      %.6f s\n", t_laplacian);
        printf("Eigenvectors:   %.6f s\n", t_eigen);
        printf("K-means:        %.6f s\n", t_kmeans);
        printf("Total time:     %.6f s\n", t_total);

    #if ENABLE_METRICS
        double score = cluster_similarity_score(S, labels, n);

        printf("\n--- Quality ---\n");
        printf("Cluster quality (intra/inter ratio): %.4f\n", score);
    #endif

        printf("===================================\n\n");

        // Append to CSV
        int write_header = 0;
        struct stat st;
        if (stat("output/performance.csv", &st) != 0) {
            write_header = 1;
        }

        FILE *pf = fopen("output/performance.csv", "a");
        if (pf) {
            if (write_header) {
                fprintf(pf,
                    "dataset,n,cols,k,clusters,mpi_procs,sigma,"
                    "t_load,t_degree,t_laplacian,t_eigen,t_kmeans,t_total,quality\n"
                );
            }
            fprintf(pf,
                "%s,%d,%d,%d,%d,%d,%.4f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                input_file, n, cols, k, clusters, size, sigma,
                t_load, t_degree, t_laplacian,
                t_eigen, t_kmeans, t_total
            );

            #if ENABLE_METRICS
            fprintf(pf, "%.6f\n", score);
            #else
            fprintf(pf, "-1\n");
            #endif

            fclose(pf);
        }
        
    }

    free(S);
    free(degree);
    if (rank == 0) free(L);
    free(U); 
    free(labels);

    MPI_Finalize();
    return 0;
}
