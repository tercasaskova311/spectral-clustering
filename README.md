# Spectral Clustering with MPI

This project implements **spectral clustering** for large datasets using **MPI parallelization**. 
It computes the similarity matrix, Laplacian, eigenvectors, and applies k-means clustering.

## Features
- Parallel computation using MPI
- Laplacian computation and eigenvector extraction
- k-means clustering in spectral embedding space
- Timing measurement for each phase
- Output CSV for benchmarking
- Optional visualization of clusters

## Structure

spectral-clustering/
├── main.c
├── laplacian.c / laplacian.h
├── eigensolver.c / eigensolver.h
├── kmeans.c / kmeans.h
├── Makefile
├── run_spectral.sh        # PBS submission script
├── data/                  # datasets
│   └── ans_batch/
│       └── test2_ddg.txt
├── output/                # CSV and plots saved here
├── README.md
├── documentation.md
├── visualization



## compilation instruction

# Load MPI module
module load mpich-3.2

# Build the code
make clean
make

## Run locally 
# Single process (debug)
mpirun -np 1 ./spectral_mpi data/ans_batch/test2_ddg.txt

# Multi-process
mpirun -np 4 ./spectral_mpi data/ans_batch/test2_ddg.txt 3 3
# arguments: <file> <k eigenvectors> <clusters>


## Running on a cluster
qsub run_spectral.sh

- output will be saved in spectral.out
- timing csv - for each part of the clustering

## References
There are multiple spectral clustering algorithms manuals, we did follow these: 


