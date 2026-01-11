#include "compute_similarity.h"
#include <math.h>

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
