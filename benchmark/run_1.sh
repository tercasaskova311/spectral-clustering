#!/bin/bash
#PBS -N spectral_1
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=8gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ
#PBS -j oe
#PBS -o benchmark/out_1.txt

cd $PBS_O_WORKDIR
module load OpenMPI/4.1.6-GCC-13.2.0

INPUT_DIR=data

for file in ${INPUT_DIR}/*.txt; do
    mpirun ./spectral_mpi "$file" 3 3
done