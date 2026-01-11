#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <mpi.h>

void load_square_matrix(const char *filename, double *S, int n, int rank);
void compute_degree_matrix(double *S, double *degree, int n, int rank, int size);
void laplacian(double *S, double *degree, double *L, int n, int rank, int size);

#endif