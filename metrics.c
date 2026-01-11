#include "metrics.h"

// Computes ratio of average intra-cluster similarity to average inter-cluster similarity
double cluster_similarity_score(double *S, int *labels, int n) {
    double intra = 0.0, inter = 0.0;
    int intra_count = 0, inter_count = 0;

    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            if(labels[i] == labels[j]){
                intra += S[i*n+j];
                intra_count++;
            } else {
                inter += S[i*n+j];
                inter_count++;
            }
        }
    }

    if(intra_count == 0) intra_count = 1; // avoid divide by 0
    if(inter_count == 0) inter_count = 1; // avoid divide by 0

    double intra_avg = intra / intra_count;
    double inter_avg = inter / inter_count;

    return intra_avg / inter_avg;
}
