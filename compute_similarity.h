#ifndef COMPUTE_SIMILARITY_H
#define COMPUTE_SIMILARITY_H

#include <mpi.h>

void compute_similarity_matrix(double *X, double *S, int n, int m, double sigma);
void load_feature_matrix(const char *filename, double *X, int n, int m);

#endif
