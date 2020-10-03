#!/bin/bash
### Begin BSUB Options
#BSUB -P CSC314
#BSUB -J o-dmda
#BSUB -W 00:20
#BSUB -nnodes 9
#BSUB -alloc_flags "smt4 gpumps"
### End BSUB Options and begin shell commands
date
module list

cpurun=0
if [ $cpurun -eq 1 ]; then
  printf "\n\nInter-node CPU-CPU dmda2d test\n"
  jsrun --smpiargs "-gpu" -n 9 -a 1 -c 42 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1 ./sf-dmda -use_gpu_aware_mpi -sf_use_default_stream
fi


gpuprofile=1
if [ $gpuprofile -eq 1 ]; then
  jsrun --smpiargs "-gpu" -n 9 -a 1 -c 42 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1 \
  nvprof -o $HOME/MEMBERWORK/dmda.%q{OMPI_COMM_WORLD_RANK}.%h.%p.nvvp \
  --profile-from-start off \
  ./sf_dmda -dm_vec_type cuda -nprof 4096
else
  printf "\n\nInter-node GPU-GPU dmda2d test with 9 nodes\n"
  jsrun --smpiargs "-gpu" -n 9 -a 1 -c 42 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1 ./sf_dmda -dm_vec_type cuda
  printf "\n\nInter-node GPU-GPU dmda2d test with 3 nodes\n"
  jsrun --smpiargs "-gpu" -n 9 -a 1 -c 7  -g 1 -r 3 -l CPU-CPU -d packed -b packed:1 ./sf_dmda -dm_vec_type cuda
fi

#  --profile-from-start off --analysis-metrics --dependency-analysis \
