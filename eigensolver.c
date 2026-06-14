#include "eigensolver.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * ScaLAPACK / BLACS Fortran interfaces.
 * This assumes the common trailing-underscore symbol convention.
 */
extern void blacs_get_(int *icontxt, int *what, int *val);
extern void blacs_gridinit_(int *icontxt, char *order, int *nprow, int *npcol);
extern void blacs_gridinfo_(int *icontxt, int *nprow, int *npcol, int *myrow, int *mycol);
extern void blacs_gridexit_(int *icontxt);
extern int  numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_(int *desc, int *m, int *n, int *mb, int *nb,
                      int *irsrc, int *icsrc, int *ictxt, int *lld, int *info);

extern void pdsyevd_(char *jobz, char *uplo,
                     int *n,
                     double *a, int *ia, int *ja, int *desca,
                     double *w,
                     double *z, int *iz, int *jz, int *descz,
                     double *work, int *lwork,
                     int *iwork, int *liwork,
                     int *info);

static int max_int(int a, int b) {
    return (a > b) ? a : b;
}

static int owner_index(int global_index, int block_size, int nprocs_dim) {
    return (global_index / block_size) % nprocs_dim;
}

static int local_index(int global_index, int block_size, int nprocs_dim) {
    int block = global_index / block_size;
    int offset = global_index % block_size;
    return (block / nprocs_dim) * block_size + offset;
}

static int global_index_from_local(int local_idx, int proc_coord,
                                   int block_size, int nprocs_dim) {
    int local_block = local_idx / block_size;
    int offset = local_idx % block_size;
    int global_block = local_block * nprocs_dim + proc_coord;
    return global_block * block_size + offset;
}

static void choose_process_grid(int size, int *nprow, int *npcol) {
    int p;
    *nprow = 1;

    /* GRID SEARCH: p is updated sequentially while scanning divisors of size.
     * OpenMP = not useful here because the loop is tiny and carries the
     * current best grid choice through *nprow.
     */
    for (p = 1; p * p <= size; p++) {
        if (size % p == 0) {
            *nprow = p;
        }
    }
    *npcol = size / (*nprow);
}

void compute_eigenvectors(double *L, double *U, int n, int k, int rank) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nb = 64;   /* ScaLAPACK block size: tune 32, 64, 128 if needed */
    int nprow, npcol;
    choose_process_grid(size, &nprow, &npcol);

    int ictxt, zero = 0, minus_one = -1;
    blacs_get_(&minus_one, &zero, &ictxt);

    char order = 'R';
    blacs_gridinit_(&ictxt, &order, &nprow, &npcol);

    int myrow, mycol, grid_nprow, grid_npcol;
    blacs_gridinfo_(&ictxt, &grid_nprow, &grid_npcol, &myrow, &mycol);

    if (myrow < 0 || mycol < 0) {
        fprintf(stderr, "Rank %d is not part of the BLACS grid\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rsrc = 0, csrc = 0;
    int local_rows = numroc_(&n, (int *)&nb, &myrow, &rsrc, &nprow);
    int local_cols = numroc_(&n, (int *)&nb, &mycol, &csrc, &npcol);
    int lld = max_int(1, local_rows);
    int local_elems = lld * local_cols;

    double *A_loc = calloc((size_t)local_elems, sizeof(double));
    double *Z_loc = calloc((size_t)local_elems, sizeof(double));
    double *w = malloc((size_t)n * sizeof(double));

    if (!A_loc || !Z_loc || !w) {
        fprintf(stderr, "Rank %d: ScaLAPACK allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int descA[9], descZ[9], info;
    descinit_(descA, &n, &n, (int *)&nb, (int *)&nb,
              &rsrc, &csrc, &ictxt, &lld, &info);
    if (info != 0) {
        fprintf(stderr, "Rank %d: descinit for A failed, info=%d\n", rank, info);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    descinit_(descZ, &n, &n, (int *)&nb, (int *)&nb,
              &rsrc, &csrc, &ictxt, &lld, &info);
    if (info != 0) {
        fprintf(stderr, "Rank %d: descinit for Z failed, info=%d\n", rank, info);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Gather each rank's BLACS coordinates and local sizes on rank 0. */
    int myinfo[5] = { myrow, mycol, local_rows, local_cols, lld };
    int *allinfo = NULL;
    if (rank == 0) {
        allinfo = malloc((size_t)5 * size * sizeof(int));
        if (!allinfo) {
            fprintf(stderr, "Rank 0: failed to allocate allinfo\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(myinfo, 5, MPI_INT, allinfo, 5, MPI_INT, 0, MPI_COMM_WORLD);

    int *sendcounts = NULL;
    int *displs = NULL;
    double *sendbuf = NULL;

    if (rank == 0) {
        if (!L) {
            fprintf(stderr, "Rank 0: L is NULL; cannot distribute matrix to ScaLAPACK\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        sendcounts = malloc((size_t)size * sizeof(int));
        displs = malloc((size_t)size * sizeof(int));
        int *coord_to_rank = malloc((size_t)nprow * npcol * sizeof(int));

        if (!sendcounts || !displs || !coord_to_rank) {
            fprintf(stderr, "Rank 0: failed to allocate distribution metadata\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* INITIALIZATION LOOP: Each iteration writes one independent entry.
         * OpenMP = parallelizable, but the array is only nprow*npcol entries
         * so parallel overhead would usually dominate.
         */
        for (int p = 0; p < nprow * npcol; p++) {
            coord_to_rank[p] = -1;
        }

        int total = 0;

        /* METADATA LOOP: sendcounts[p] and coord_to_rank[...] are independent
         * writes, but displs[p] depends on the prefix sum stored in total.
         * OpenMP = not directly parallelizable without a parallel prefix-sum
         * phase; this setup cost is small compared with the eigensolve.
         */
        for (int p = 0; p < size; p++) {
            int prow = allinfo[5 * p + 0];
            int pcol = allinfo[5 * p + 1];
            int prows = allinfo[5 * p + 2];
            int pcols = allinfo[5 * p + 3];
            int plld = allinfo[5 * p + 4];

            coord_to_rank[prow * npcol + pcol] = p;
            sendcounts[p] = plld * pcols;
            displs[p] = total;
            total += sendcounts[p];

            (void)prows; /* prows is informational; lld is what determines storage */
        }

        sendbuf = calloc((size_t)total, sizeof(double));
        if (!sendbuf) {
            fprintf(stderr, "Rank 0: failed to allocate ScaLAPACK send buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Pack dense row-major C matrix L into ScaLAPACK local column-major blocks. */
        /* PACKING LOOP: Each (gi, gj) element of L maps to exactly one
         * block-cyclic destination in sendbuf. There are no algorithmic
         * dependencies between matrix elements, but parallel writes would
         * target different regions of one shared sendbuf.
         * OpenMP = parallelizable over gi/gj if the destination mapping is
         * guaranteed unique, as it is here. The write indices are indirect,
         * so this should be tested carefully for cache behavior and false
         * sharing before using it for performance.
         */
        for (int gj = 0; gj < n; gj++) {
            int pcol = owner_index(gj, nb, npcol);
            int lj = local_index(gj, nb, npcol);

            for (int gi = 0; gi < n; gi++) {
                int prow = owner_index(gi, nb, nprow);
                int li = local_index(gi, nb, nprow);
                int p = coord_to_rank[prow * npcol + pcol];
                int plld = allinfo[5 * p + 4];
                sendbuf[displs[p] + li + lj * plld] = L[(size_t)gi * n + gj];
            }
        }

        free(coord_to_rank);
    }

    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_DOUBLE,
                 A_loc, local_elems, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(sendbuf);
        free(sendcounts);
        free(displs);
        free(allinfo);
    }

    char jobz = 'V';
    char uplo = 'U';
    int ia = 1, ja = 1, iz = 1, jz = 1;

    int lwork = -1;
    int liwork = -1;
    double work_query = 0.0;
    int iwork_query = 0;

    pdsyevd_(&jobz, &uplo,
             &n,
             A_loc, &ia, &ja, descA,
             w,
             Z_loc, &iz, &jz, descZ,
             &work_query, &lwork,
             &iwork_query, &liwork,
             &info);

    if (info != 0) {
        fprintf(stderr, "Rank %d: PDSYEVD workspace query failed, info=%d\n", rank, info);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    lwork = (int)work_query;
    liwork = iwork_query;

    double *work = malloc((size_t)lwork * sizeof(double));
    int *iwork = malloc((size_t)liwork * sizeof(int));

    if (!work || !iwork) {
        fprintf(stderr, "Rank %d: failed to allocate PDSYEVD workspace\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    pdsyevd_(&jobz, &uplo,
             &n,
             A_loc, &ia, &ja, descA,
             w,
             Z_loc, &iz, &jz, descZ,
             work, &lwork,
             iwork, &liwork,
             &info);

    if (info != 0) {
        fprintf(stderr, "Rank %d: PDSYEVD failed, info=%d\n", rank, info);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Gather only the first k eigenvectors into row-major U on rank 0. */
    int local_pairs = 0;

    /* COUNT LOOP: local_pairs is a reduction-style counter over local
     * ScaLAPACK rows and columns. Z_loc is read-only here.
     * OpenMP = parallelizable with a reduction on local_pairs, but this loop
     * only covers the local part of the first k eigenvectors.
     */
    for (int lj = 0; lj < local_cols; lj++) {
        int gj = global_index_from_local(lj, mycol, nb, npcol);
        if (gj >= k || gj >= n) continue;

        for (int li = 0; li < local_rows; li++) {
            int gi = global_index_from_local(li, myrow, nb, nprow);
            if (gi < n) local_pairs++;
        }
    }

    int local_doubles = 3 * local_pairs;
    double *local_pack = malloc((size_t)max_int(1, local_doubles) * sizeof(double));
    if (!local_pack) {
        fprintf(stderr, "Rank %d: failed to allocate eigenvector gather pack\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int pos = 0;

    /* LOCAL PACK LOOP: Each selected local eigenvector entry is packed as
     * (global row, global column, value). The variable pos is a loop-carried
     * write index, so iterations depend on the previous number of accepted
     * entries.
     * OpenMP = not directly parallelizable with the current append pattern.
     * A parallel version would need precomputed offsets or thread-private
     * buffers followed by a merge.
     */
    for (int lj = 0; lj < local_cols; lj++) {
        int gj = global_index_from_local(lj, mycol, nb, npcol);
        if (gj >= k || gj >= n) continue;

        for (int li = 0; li < local_rows; li++) {
            int gi = global_index_from_local(li, myrow, nb, nprow);
            if (gi >= n) continue;

            local_pack[pos++] = (double)gi;
            local_pack[pos++] = (double)gj;
            local_pack[pos++] = Z_loc[li + lj * lld];
        }
    }

    int *recvcounts = NULL;
    int *rdispls = NULL;
    double *recvbuf = NULL;

    if (rank == 0) {
        recvcounts = malloc((size_t)size * sizeof(int));
        rdispls = malloc((size_t)size * sizeof(int));
        if (!recvcounts || !rdispls) {
            fprintf(stderr, "Rank 0: failed to allocate gather metadata\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&local_doubles, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_doubles = 0;

        /* DISPLACEMENT LOOP: rdispls[p] is a prefix sum over recvcounts.
         * OpenMP = not directly parallelizable because total_doubles carries
         * the cumulative offset from one iteration to the next.
         */
        for (int p = 0; p < size; p++) {
            rdispls[p] = total_doubles;
            total_doubles += recvcounts[p];
        }

        recvbuf = malloc((size_t)max_int(1, total_doubles) * sizeof(double));
        if (!recvbuf) {
            fprintf(stderr, "Rank 0: failed to allocate eigenvector gather buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(local_pack, local_doubles, MPI_DOUBLE,
                recvbuf, recvcounts, rdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* UNPACK LOOP: Each packed triple writes one U[gi, gj] entry. The
         * ScaLAPACK ownership mapping should make entries unique across
         * ranks, so there is no intended write conflict.
         * OpenMP = parallelizable over the received triples if uniqueness is
         * preserved. The nested p/q structure and indirect indexing mean it
         * may not be a major performance target compared with PDSYEVD.
         */
        for (int p = 0; p < size; p++) {
            for (int q = rdispls[p]; q < rdispls[p] + recvcounts[p]; q += 3) {
                int gi = (int)recvbuf[q];
                int gj = (int)recvbuf[q + 1];
                double val = recvbuf[q + 2];
                U[(size_t)gi * k + gj] = val;
            }
        }
        free(recvbuf);
        free(recvcounts);
        free(rdispls);
    }

    MPI_Bcast(U, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(local_pack);
    free(work);
    free(iwork);
    free(A_loc);
    free(Z_loc);
    free(w);

    blacs_gridexit_(&ictxt);
}
