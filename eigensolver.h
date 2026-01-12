#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H

/*
* Compute the first k eigenvectors of the Laplacian.
* The computation is performed on rank 0 using LAPACK (dsyev), and the
* resulting eigenvectors are broadcast to all MPI ranks.
*/
void compute_eigenvectors(double *L, double *U, int n, int k, int rank);

#endif