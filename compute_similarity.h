#ifndef COMPUTE_SIMILARITY_H
#define COMPUTE_SIMILARITY_H

#include <mpi.h>

/*
* Compute an RBF similarity matrix from a feature matrix X.
* If *sigma <= 0, the bandwidth is automatically selected using
* the median heuristic and written back through the pointer.
*/
void compute_similarity_matrix(double *X, double *S, int n, int m, double *sigma);

/* Load an n x m feature matrix from file. */
void load_feature_matrix(const char *filename, double *X, int n, int m);

#endif
