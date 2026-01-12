#ifndef KMEANS_H
#define KMEANS_H

/*
* Distributed k-means clustering on the spectral embedding U.
* Each MPI rank processes a subset of rows and participates in global
* reductions to update cluster centroids.
*/
void kmeans(double *U, int n, int k, int clusters, int iters,
            int rank, int size, int *labels);

#endif