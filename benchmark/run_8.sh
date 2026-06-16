#!/bin/bash
#PBS -N spectral_8
#PBS -l select=1:ncpus=8:mpiprocs=8:mem=8gb
#PBS -l walltime=00:10:00
#PBS -q shortCPUQ
#PBS -j oe
#PBS -o benchmark/out_8.txt

cd $PBS_O_WORKDIR
module load OpenMPI/4.1.5-GCC-12.3.0
module load ScaLAPACK/2.2.0-gompi-2023a-fb 

INPUT_DIR=data/input

for file in ${INPUT_DIR}/*.txt; do
    mpirun --hostfile $PBS_NODEFILE ./spectral_mpi "$file" 3 3
done