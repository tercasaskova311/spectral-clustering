# Parallel Spectral Clustering with MPI

This project implements **spectral clustering** using **MPI parallelization** on a distributed-memory HPC system. 
The goal is to study the performance and scalability of spectral clustering while preserving clustering quality.

The implementation supports both:
- **Precomputed similarity matrices** (square input)
- **Feature matrices**, for which a similarity matrix is computed using an RBF kernel

---

## Algorithm Overview

The implemented spectral clustering pipeline follows these steps:

1. Load input data (similarity matrix or feature matrix)
2. If needed, compute a similarity matrix using an RBF kernel
3. Construct the graph Laplacian
4. Compute the first *k* eigenvectors of the Laplacian
5. Run k-means clustering on the spectral embedding
6. Optionally evaluate clustering quality
7. Record detailed timing information for benchmarking

---

## Parallelization Strategy

- **MPI** is used to parallelize:
  - Degree computation
  - Laplacian construction
  - k-means clustering
- **Eigen-decomposition** is performed **serially on rank 0** using LAPACK (`dsyev`) and broadcast to all ranks.
  - This design choice was imposed by the available HPC environment, which did not provide distributed eigensolvers.
- Communication is handled using `MPI_Bcast` and `MPI_Allreduce`.
- Large matrices are freed as early as possible to limit memory usage.

---

## Project Structure

```
spectral-clustering/
├── main.c                  # Main driver and pipeline
├── laplacian.c / .h        # Matrix loading, degree, Laplacian
├── eigensolver.c / .h      # LAPACK eigen-decomposition
├── kmeans.c / .h           # Distributed k-means
├── compute_similarity / .h # RBF similarity + sigma heuristic
├── read_matrix_size.c / .h # Input format detection
├── metrics.c / .h          # Clustering quality metric
├── Makefile
├── run_spectral.sh         # General PBS job submission script
├── data/                   # Input datasets
├── output/                 # CSV benchmarking results
├── benchmark/              # PBS job submission scripts for benchmarks
```

---

## Clustering Quality Metric

An optional quality metric is implemented:

- Ratio of **average intra-cluster similarity** to **average inter-cluster similarity**
- Values greater than 1 indicate well-separated clusters
- Enabled by default, but can be disabled at compile time for pure benchmarking

---

## Compilation

### Load required modules (example)

```bash 
module load mpich-3.2
```

### Build 

```bash
make clean
make
```

### Build without quality metrics

```bash
make clean
make CFLAGS="-std=c99 -Wall -Wextra -O3 -DENABLE_METRICS=0"
```

---

## Running locally 

```bash
# Single process (debug)
mpirun.actual -np 1 ./spectral_mpi data/ans_batch/test2_ddg.txt

# Multiple processes
mpirun.actual -np 4 ./spectral_mpi data/ans_batch/test2_ddg.txt 3 3
```

### Arguments:

```
./spectral_mpi <input_file> <k_eigenvectors> <clusters> <sigma>
```

If `sigma` is omitted or negative, it is automatically selected using the median heuristic.

---

## Running on the Cluster (PBS)
Submit the job using:

```bash
qsub run_spectral.sh
```

- Standard output is written to the PBS output file 
- Performance results are appended to `output/performance.csv`

---

## Benchmarking Output
Benchmarking experiments were performed using PBS batch scripts, with one script per MPI configuration. For each MPI configuration, the program is run sequentially on all datasets contained in the `data/` directory.

For each run, the following timings are recorded:

- Similarity computation / matrix loading
- Degree computation
- Laplacian construction
- Eigen-decomposition
- k-means clustering
- Total runtime

Each execution appends timing and quality information to a shared CSV file, enabling strong-scaling and dataset-size analysis without manual intervention.

---

## Dependencies

- MPICH
- BLAS
- LAPACK

## References
- von Luxburg, U. (2007). A Tutorial on Spectral Clustering. Statistics and Computing, 17(4), 395-416. Available at: https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf

- Eizenman, N. Spectral Clustering using C and Python. GitHub repository. https://github.com/nir-eizenman/Spectral-Clustering-using-C-and-Python



