#!/bin/bash
#PBS -l select=2:ncpus=8:mem=8gb

# set max execution time
#PBS -l walltime=00:10:00

# Queue
#PBS -q short_cpuQ

# output files
#PBS -j oe
#PBS -o spectral.out

cd $PBS_O_WORKDIR
module load mpich-3.2

#initialize csv
mkdir -p output
echo "dataset,n,clusters,procs,t_load,t_degree,t_laplacian,t_eigen,t_kmeans,total" > output/timing.csv

mpirun.actual -np 16 ./spectral_mpi


