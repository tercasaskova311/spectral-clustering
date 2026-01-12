#!/bin/bash
#PBS -N spectral_8
#PBS -l select=2:ncpus=8:mem=8gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ
#PBS -j oe
#PBS -o benchmark/out_8.txt

cd $PBS_O_WORKDIR
module load mpich-3.2

INPUT_DIR=data

for file in ${INPUT_DIR}/*.txt; do
    mpirun.actual -np 8 ./spectral_mpi "$file" 3 3
done