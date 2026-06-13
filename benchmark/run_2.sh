#!/bin/bash
#PBS -N spectral_2
#PBS -l select=1:ncpus=2:mpiprocs=2:mem=8gb
#PBS -l walltime=00:10:00
#PBS -q shortCPUQ
#PBS -j oe
#PBS -o benchmark/out_2.txt

cd $PBS_O_WORKDIR
module load OpenMPI/4.1.5-GCC-12.3.0

INPUT_DIR=data/input

for file in ${INPUT_DIR}/*.txt; do
    mpirun --hostfile $PBS_NODEFILE ./spectral_mpi "$file" 3 3
done