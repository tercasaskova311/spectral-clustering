#include "kmeans.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

void kmeans(double *U, int n, int k, int clusters, int iters, int rank, int size, int *labels){
    /* Allocate and initialize centroids on rank 0 */

    /* STRATEGY: centroids (write) - no dependency between different c
    * but WAW and RAW dependencies on centroids[c * k + j] within the same c and j.
     * OpenMP = not parallelizable since centroids is shared with dependencies.
     */
    double *centroids = malloc(clusters * k * sizeof(double));
    if (rank == 0){
        for (int c = 0; c < clusters; c++)
            for (int j = 0; j < k; j++)
                centroids[c * k + j] = U[c * k + j];
    }

    MPI_Bcast(centroids, clusters * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* Determine row ownership */
    int rows = n / size;
    int rem = n % size;
    int start = (rank < rem) ? rank * (rows + 1) : rank * rows + rem;
    int end = start + ((rank < rem) ? rows + 1 : rows);

    /* OUTER LOOP: Iterate for a fixed number of times 
     * STRATEGY: labels[i] (write) - no dependency between different i, 
     *.          but WAW and RAW dependencies on labels[i] within the same i.
     *           centroids (read) - no dependency
     * OpenMP = parallelizable over local rows. Each thread computes its own portion 
     * of the labels.
     */

    for (int it = 0; it < iters; it++) {
        double *local_sum = calloc(clusters * k, sizeof(double));
        int *local_count = calloc(clusters, sizeof(int));

        /* Assignment step */
        /* STRATEGY: labels[i] (write) - no dependency between different i, 
         *.          but WAW and RAW dependencies on labels[i] within the same i.
         *           centroids (read) - no dependency
         * OpenMP = parallelizable over local rows. Each thread computes its own portion 
         * of the labels.
         */
        for (int i = start; i < end; i++) {
            int best = 0;
            double best_dist = 1e100;

            /* Compute distances to each centroid 
             * STRATEGY: best_dist (write) - no dependency between different c,
             *           but WAW and RAW dependencies on best_dist within the same c.
             *          centroids (read) - no dependency
             * OpenMP = parallelizable over clusters. 
             * Each thread computes its own portion of the distances.
            */
            for (int c = 0; c < clusters; c++) {
                double dist = 0.0;
                for (int j = 0; j < k; j++) {
                    double d = U[i * k + j] - centroids[c * k + j];
                    dist += d * d;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            
            /* Assign the point to the closest centroid */
            /* STRATEGY: labels[i] (write) - no dependency between different i, 
             *.          but WAW and RAW dependencies on labels[i] within the same i.
             *           centroids (read) - no dependency
             * OpenMP = parallelizable over local rows. Each thread computes its own portion 
             * of the labels.
             */
            labels[i] = best;
            local_count[best]++;
            for (int j = 0; j < k; j++) 
                local_sum[best * k + j] += U[i * k + j];
        }

        /* Global reduction of centroid updates */
        MPI_Allreduce(MPI_IN_PLACE, local_sum, 
                      clusters * k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, local_count,
                      clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        /* Update centroids */
        /* STRATEGY: centroids[c * k + j] (write) - no dependency between different c and j,
         *           but WAW and RAW dependencies on centroids[c * k + j] within the same c and j.
         *           local_sum (read) - no dependency
         *           local_count (read) - no dependency
         * OpenMP = parallelizable over clusters and dimensions. Each thread computes its own portion 
         * of the centroids.
         */
        for (int c = 0; c < clusters; c++) {
            if (local_count[c] > 0) 
                for (int j = 0; j < k; j++) 
                    centroids[c * k + j] = local_sum[c * k + j] / local_count[c];
        }

        free(local_sum); 
        free(local_count);
    }
    
    free(centroids);
}