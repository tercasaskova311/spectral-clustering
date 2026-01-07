void kmeans(double *U, int n, int k, int clusters, int max_iter, int rank, int size, int *labels){
    double *centroids = malloc(clusters  k * sizeof(double));

    if (rank == 0){
        for (int c = 0; c < clusters; c++){
            for (int j = 0; j < k; j++){
                centroids[c * k + j] = U[c * k + j];
            }
        }
    }

    MPI_Bcast(centroids, clusters, *k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int iter = 0; iter < max_iter; iter++) {

        double *local_sum = calloc(clusters * k, sizeof(double));
        int *local_count = calloc(clusters, sizeof(int));

        // same row partition logic as before
        int rows_per_proc = n / size;
        int start = (rank < n % size)
                ? rank * (rows_per_proc + 1)
                : rank * rows_per_proc + (n % size);
        int end = start + ((rank < n % size) ? rows_per_proc + 1 : rows_per_proc);

        // assignment step
        for (int i = start; i < end; i++) {
            int best = 0;
            double best_dist = 1e100;

            for (int c = 0; c < clusters; c++) {
                double dist = 0.0;
                for (int j = 0; j < k; j++) {
                    double diff = U[i * k + j] - centroids[c * k + j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }

            labels[i] = best;
            local_count[best]++;
            for (int j = 0; j < k; j++) {
                local_sum[best * k + j] += U[i * k + j];
            }
        }

        // global reduction
        MPI_Allreduce(MPI_IN_PLACE, local_sum,
                    clusters * k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, local_count,
                    clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // update centroids
        for (int c = 0; c < clusters; c++) {
            if (local_count[c] > 0) {
                for (int j = 0; j < k; j++) {
                    centroids[c * k + j] =
                        local_sum[c * k + j] / local_count[c];
                }
            }
        }

        free(local_sum);
        free(local_count);
    }
}