#include "compute_similarity.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void compute_similarity_matrix(double *X, double *S, int n, int m, double sigma){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            double dist2 = 0.0;
            for(int k=0;k<m;k++){
                double diff = X[i*m + k] - X[j*m + k];
                dist2 += diff*diff;
            }
            S[i*n + j] = exp(-dist2/(2.0*sigma*sigma));
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
                fprintf(stderr, "Error reading feature matrix");
                fclose(f);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    fclose(f);
}