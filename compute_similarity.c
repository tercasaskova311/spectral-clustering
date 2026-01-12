#include "compute_similarity.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
* Compute the RBF bandwidth using the median heuristic.
* A subset of pairwise distances is sampled to avoid O(n^2) cost
* for large datasets.
*/
double compute_sigma_median_heuristic(double *X, int n, int m) {
    int max_samples = (n < 100) ? n * (n - 1) / 2 : 5000;
    double *distances = malloc(max_samples * sizeof(double));
    int count = 0;
    
    for (int i = 0; i < n && count < max_samples; i++) {
        int step = (n > 100) ? n / 50 : 1; 
        for (int j = i + 1; j < n && count < max_samples; j += step) {
            double dist2 = 0.0;
            for (int k = 0; k < m; k++) {
                double diff = X[i*m + k] - X[j*m + k];
                dist2 += diff * diff;
            }
            distances[count++] = sqrt(dist2);
        }
    }
    
    /* Sort sampled distances */
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                double tmp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = tmp;
            }
        }
    }
    
    double median = distances[count / 2];
    free(distances);
    
    return (median > 0.0) ? median : 1.0;
}

void compute_similarity_matrix(double *X, double *S, int n, int m, double *sigma){
    /* Automatically determine sigma if not provided */
    if (*sigma <= 0.0) {
        *sigma = compute_sigma_median_heuristic(X, n, m);
    }
    
    double two_sigma_sq = 2.0 * (*sigma) * (*sigma);
    
    /* Compute full similarity matrix */
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if (i == j) {
                S[i * n + j] = 1.0;
                continue;
            }
            double dist2 = 0.0;
            for (int k = 0; k < m; k++){
                double diff = X[i * m + k] - X[j * m + k];
                dist2 += diff * diff;
            }
            S[i * n + j] = exp(-dist2 / two_sigma_sq);
        }
    }
}

void load_feature_matrix(const char *filename, double *X, int n, int m) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("fopen");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (fscanf(f, " %lf%*[, ]", &X[i*m + j]) != 1) {
                fprintf(stderr, "Error reading feature matrix at row %d, col %d\n", i, j);
                fclose(f);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    fclose(f);
}