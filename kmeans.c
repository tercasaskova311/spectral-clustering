#include "kmeans.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

void kmeans(double *U, int n, int k, int clusters, int iters, int rank, int size, int *labels){
    /* Allocate and initialize centroids on rank 0 */
    /* Each iteration initializes a unique centroid element.
     * No loop-carried dependencies exist. The loop is OpenMP parallelizable.
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
     * Each iteration initializes a unique centroid element.
     * No loop-carried dependencies exist.
     * OpenMP = parallelizable over local rows. Each thread computes 
     * its own portion of the labels and centroid updates.
     */

    for (int it = 0; it < iters; it++) {
        double *local_sum = calloc(clusters * k, sizeof(double));
        int *local_count = calloc(clusters, sizeof(int));

        /* Assignment step
         *Label assignment is independent across points. 
         However, updates to local_sum and local_count create 
         race conditions under OpenMP and require reductions, 
         atomics, or thread-private accumulators.
         */
        for (int i = start; i < end; i++) {
            int best = 0;
            double best_dist = 1e100;

            /* Compute distances to each centroid 
             * Reduction dependency (minimum search) 
             * through best_dist and associated index best.
             * The loop contains a min-reduction dependency 
             * and is usually left sequential. Parallelizing 
             * the outer point loop provides greater benefit.
            */
            for (int c = 0; c < clusters; c++) {
                double dist = 0.0;
                for (int j = 0; j < k; j++) {
                    /* reduction dependency (j => j+1) - but usually k is small */
                    double d = U[i * k + j] - centroids[c * k + j];
                    dist += d * d;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            
            /* Assign the point to the closest centroid */
            /* OpenMP parallelization of the assignment loop requires thread-private 
             * partial sums and counts (or atomics/reductions) because 
             * multiple points may contribute to the same cluster simultaneously.
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
        /* Each centroid component is updated independently from globally reduced 
         *sums and counts. No loop-carried dependencies exist.
         * OpenMP = parallelizable over clusters and dimensions. Each thread computes 
         * its own portion of the centroids.
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