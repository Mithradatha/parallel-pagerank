#!/bin/bash
#
## SLURM submission script for OpenMP PageRank Job
#SBATCH --job-name=pagerank
#SBATCH --output=pagerank.%J.out
#SBATCH --error=pagerank.%J.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00
#SBATCH --partition=class

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
## change n to modify number of pages in ranking
## change q to modify damping factor
## change k to modify number of matvec iterations
./sparse_omp.c
