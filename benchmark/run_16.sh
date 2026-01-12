#!/bin/bash
#PBS -N spectral_16
#PBS -l select=2:ncpus=8:mem=8gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ
#PBS -j oe
#PBS -o benchmark/out_16.txt

cd $PBS_O_WORKDIR
module load mpich-3.2

mpirun.actual -np 16 ./spectral_mpi data/ans_batch/test2_ddg.txt 3 3