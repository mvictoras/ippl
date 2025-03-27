#!/bin/bash
module --force purge
module load Stages/2025
module load GCC
module load ParaStationMPI
module load CMake
module load CUDA

export PENNINGTRAP_BINDIR=/p/project1/jureap1/sewell1/ippl/build/alpine
srun --ntasks-per-node=1 --cpu-bind=cores --export=ALL,OMP_PROC_BIND=spread,OMP_PLACES=threads $PENNINGTRAP_BINDIR/PenningTrap 16 16 16 4096 20 FFT 0.01 LeapFrog --info 5
