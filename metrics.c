#include "metrics.h"

/*
* Compute the ratio between average intra-cluster similarity
* and average inter-cluster similarity. Values > 1 indicate
* well-separated clusters.
*/
double cluster_similarity_score(double *S, int *labels, int n) {
    double intra = 0.0, inter = 0.0;
    int intra_count = 0, inter_count = 0;

    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if(labels[i] == labels[j]){
                intra += S[i * n + j];
                intra_count++;
            } else {
                inter += S[i * n + j];
                inter_count++;
            }
        }
    }

    if(intra_count == 0) intra_count = 1; 
    if(inter_count == 0) inter_count = 1; 
    
    return (intra / intra_count) / (inter / inter_count);
}
