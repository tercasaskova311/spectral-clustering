#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H

/*
* Compute the first k eigenvectors of the Laplacian.
* The computation is distributed with ScaLAPACK/BLACS (pdsyevd).
* Rank 0 packs the dense Laplacian into block-cyclic layout, gathers the
* selected eigenvectors, and broadcasts the spectral embedding to all MPI ranks.
*/
void compute_eigenvectors(double *L, double *U, int n, int k, int rank);

#endif
