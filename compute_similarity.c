#include "compute_similarity.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
* compute_similarity_matrix()  —  O(n²m) RBF kernel computation
*/

double compute_sigma_median_heuristic(double *X, int n, int m) {
    int max_samples = (n < 100) ? n * (n - 1) / 2 : 5000;
    double *distances = malloc(max_samples * sizeof(double));
    int count = 0;
    /*
     *SAMPLING LOOP: The loops carry a shared 'count', index used to write into 'distances'. 
     *STRATEGY: count (read/write)  - dependency RAW, WAW, WAR
     *          distances[] (write) - dependency WAW
     *          X (read)          - no dependency
     *Open MP = iterations are not independent due to shared 'count' and 'distances' state.
     *therefore no openmp parallelisation possible in this loop.
     */
    
    for (int i = 0; i < n && count < max_samples; i++) {
        int step = (n > 100) ? n / 50 : 1; 
        for (int j = i + 1; j < n && count < max_samples; j += step) {

            /*
             * DISTANCE LOOP: Pure reduction — each iteration reads independent
             * elements X[i*m+k] and X[j*m+k] (no cross-iteration dependency)
             * and accumulates into dist2.
             * STRATEGY: dist2 - RAW 
             *           X - read only
             *           diff - private  
             * Open MP = parallelizable, no WAR or WAW,
             * since iterations are independent and dist2 is private to each (i,j) pair.
             */ 
            double dist2 = 0.0;
            for (int k = 0; k < m; k++) {
                double diff = X[i*m + k] - X[j*m + k];
                dist2 += diff * diff;
            }
            distances[count++] = sqrt(dist2);
        }
    }

    /*
     * BUBBLE SORT: loop iteration j+1 depends on val. modified by j
     * STRATEGY: distances[] - dependency WAW, RAW, WAR
     * OpenMP = not parallelisable due to loop-carried dependencies on distances[] during sorting.
     */

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
    
    /* OUTER LOOP: this is the outer loop for computing the similarity matrix 
     * STRATEGY: S[i*n+j] (write) - dependency WAW
     *           X (read)       - no dependency
     *           sigma (read)    - no dependency
     * OpenMP = perfect candidate for parallelisation since each S[i*n+j] is independent 
     * and X and sigma are read-only.
     */
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){

            /*INNER LOOP: this is the inner loop for computing the similarity matrix 
            *STRATEGY: S[i *n+j] (write) - dependency WAW
            *Open MP = also parallelisable since each S[i*n+j] is independent and there are no dependencies on X or sigma. However, since this is a simple assignment for the diagonal, we can skip the inner loop entirely for i==j and directly set S[i*n+i] = 1.0 without parallelisation.
            */
            if (i == j) {
                S[i * n + j] = 1.0;
                continue;
            }

            /*DISTANCE LOOP: this is the loop for computing the squared Euclidean distance between points i and j 
             * STRATEGY: dist2 (write) - dependency RAW reduction
             * OpenMP = parallelisable since each dist2 is private to each (i,j) pair and iterations of the k-loop are independent. But probably not worth parallelising this inner loop due to the overhead of parallelisation and the fact that m is typically small compared to n. 
             */
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