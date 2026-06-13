#!/bin/bash
#PBS -N spectral_4
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=8gb
#PBS -l walltime=00:10:00
#PBS -q shortCPUQ
#PBS -j oe
#PBS -o benchmark/out_4.txt

cd $PBS_O_WORKDIR
module load OpenMPI/4.1.6-GCC-13.2.0

INPUT_DIR=data/input

for file in ${INPUT_DIR}/*.txt; do
    mpirun --hostfile $PBS_NODEFILE ./spectral_mpi "$file" 3 3
done