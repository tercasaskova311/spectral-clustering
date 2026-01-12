#ifndef METRICS_H
#define METRICS_H

/*
* Compute a simple clustering quality metric:
* ratio of average intra-cluster similarity to average inter-cluster similarity.
*/
double cluster_similarity_score(double *S, int *labels, int n);

#endif
