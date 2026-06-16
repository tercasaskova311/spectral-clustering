#!/bin/bash
#PBS -l select=2:ncpus=8:mpiprocs=8:mem=8gb

# set max execution time
#PBS -l walltime=00:10:00

# Queue
#PBS -q shortCPUQ

# output files
#PBS -j oe
#PBS -o spectral.out

cd $PBS_O_WORKDIR
module load OpenMPI/4.1.6-GCC-13.2.0
module load ScaLAPACK/2.2.0-gompi-2023a-fb 

mkdir -p output

mpirun --hostfile $PBS_NODEFILE ./spectral_mpi data/input/matrix_n2000.txt 3 3

