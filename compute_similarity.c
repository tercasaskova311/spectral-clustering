#include "compute_similarity.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Auto-compute sigma using median heuristic
double compute_sigma_median_heuristic(double *X, int n, int m) {
    // Sample pairwise distances (limit to avoid O(n²) for large n)
    int max_samples = (n < 100) ? n * (n - 1) / 2 : 5000;
    double *distances = malloc(max_samples * sizeof(double));
    int count = 0;
    
    for (int i = 0; i < n && count < max_samples; i++) {
        int step = (n > 100) ? n / 50 : 1;  // Sample every step-th point
        for (int j = i + 1; j < n && count < max_samples; j += step) {
            double dist2 = 0.0;
            for (int k = 0; k < m; k++) {
                double diff = X[i*m + k] - X[j*m + k];
                dist2 += diff * diff;
            }
            distances[count++] = sqrt(dist2);
        }
    }
    
    // Find median distance
    // Simple bubble sort for small arrays, good enough for sampling
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                double temp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp;
            }
        }
    }
    
    double median_dist = distances[count / 2];
    free(distances);
    
    // Sigma = median_distance (common heuristic)
    return median_dist > 0.0 ? median_dist : 1.0;
}

void compute_similarity_matrix(double *X, double *S, int n, int m, double sigma){
    // If sigma not provided (<=0), compute automatically
    if (sigma <= 0.0) {
        sigma = compute_sigma_median_heuristic(X, n, m);
        printf("Auto-computed sigma: %.4f\n", sigma);
    }
    
    double two_sigma_sq = 2.0 * sigma * sigma;
    
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if (i == j) {
                S[i*n + j] = 1.0;  // Self-similarity = 1
                continue;
            }
            
            double dist2 = 0.0;
            for(int k=0;k<m;k++){
                double diff = X[i*m + k] - X[j*m + k];
                dist2 += diff*diff;
            }
            S[i*n + j] = exp(-dist2/two_sigma_sq);
        }
    }
}

// Separate loader for feature matrices (n x m)
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
    printf("Loaded feature matrix: %d x %d\n", n, m);
}