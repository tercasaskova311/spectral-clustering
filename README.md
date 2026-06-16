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
- **Eigen-decomposition** is performed with **ScaLAPACK/BLACS** using `pdsyevd`.
  - Rank 0 packs the dense Laplacian into ScaLAPACK's block-cyclic layout.
  - The first *k* eigenvectors are gathered back into a row-major spectral embedding and broadcast to all ranks.
- Communication is handled using `MPI_Bcast`, `MPI_Gather`, `MPI_Scatterv`, `MPI_Gatherv`, and `MPI_Allreduce`.
- Large matrices are freed as early as possible to limit memory usage, but the full similarity matrix is still stored on each rank.

---

## Project Structure

```text
spectral-clustering/
|-- main.c                    # Main driver and pipeline
|-- laplacian.c / .h          # Matrix loading, degree, Laplacian
|-- eigensolver.c / .h        # ScaLAPACK eigen-decomposition
|-- eigensolver_serial.c      # Deprecated serial LAPACK reference implementation
|-- kmeans.c / .h             # Distributed k-means
|-- compute_similarity.c / .h # RBF similarity + sigma heuristic
|-- read_matrix_size.c / .h   # Input format detection
|-- metrics.c / .h            # Clustering quality metric
|-- Makefile
|-- run_spectral.sh           # General PBS job submission script
|-- data/                     # Input datasets
|-- img/                      # Strong/weak scaling plots
|-- output/                   # CSV benchmarking results
`-- benchmark/                # PBS job submission scripts for benchmarks
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
module load OpenMPI/4.1.5-GCC-12.3.0
module load ScaLAPACK/2.2.0-gompi-2023a-fb
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

## Running Locally

```bash
# Single process (debug)
mpirun -np 1 ./spectral_mpi data/input/matrix_n500.txt

# Multiple processes
mpirun -np 4 ./spectral_mpi data/input/matrix_n500.txt 3 3
```

### Arguments

```text
./spectral_mpi [input_file] [k_eigenvectors] [clusters] [sigma]
```

If `sigma` is omitted or negative for feature-matrix input, it is automatically selected using the median heuristic.
For square similarity-matrix input, the sigma value is recorded but not used.

---

## Running on the Cluster (PBS)

Submit the job using:

```bash
qsub run_spectral.sh
```

- Standard output is written to the PBS output file
- The general script runs `data/input/matrix_n2000.txt` with `k = 3` and `clusters = 3`
- Performance results are appended to `output/performance_1.csv`

---

## Benchmarking Output

Benchmarking experiments were performed using PBS batch scripts, with one script per MPI configuration.
For each MPI configuration, the program is run sequentially on all datasets contained in the `data/input/` directory.

For each run, the following timings are recorded:

- Similarity computation / matrix loading
- Degree computation
- Laplacian construction
- Eigen-decomposition
- k-means clustering
- Total runtime

Each execution appends timing and quality information to a shared CSV file, enabling strong-scaling and dataset-size analysis without manual intervention.
The `img/` directory contains generated strong-scaling and weak-scaling plots, including comparisons for the ScaLAPACK eigensolver.

---

## Dependencies

- MPI implementation such as OpenMPI or MPICH
- BLAS
- LAPACK
- ScaLAPACK
- gfortran runtime

## References

- von Luxburg, U. (2007). A Tutorial on Spectral Clustering. Statistics and Computing, 17(4), 395-416. Available at: https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf
- Eizenman, N. Spectral Clustering using C and Python. GitHub repository. https://github.com/nir-eizenman/Spectral-Clustering-using-C-and-Python
