#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <mpi.h>

/* Load a square similarity matrix from file. */
void load_square_matrix(const char *filename, double *S, int n, int rank);

/* Compute the degree vector of the similarity matrix in parallel. */
void compute_degree_matrix(double *S, double *degree, int n, int rank, int size);

/* Construct the graph Laplacian L = D - S in parallel. */
void laplacian(double *S, double *degree, double *L, int n, int rank, int size);

#endif