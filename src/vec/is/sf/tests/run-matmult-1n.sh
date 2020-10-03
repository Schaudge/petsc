#!/bin/bash
### Begin BSUB Options
#BSUB -P CSC314
#BSUB -J o-matmult-1n
#BSUB -W 00:20
#BSUB -nnodes 1
#BSUB -alloc_flags "smt4"
##BSUB -alloc_flags "smt4 gpumps"
### End BSUB Options and begin shell commands

printf "\n\nIntra-node GPU matmult with 6 ranks + 6 GPUs\n"

profile=0

#for matrix in HV15R inline_1 Emilia_923 oilpan
for matrix in HV15R
do
if [ $profile -eq 1 ]; then
  echo "Run with profiling\n"
  jsrun --smpiargs "-gpu" -n 6 -a 1 -c 2 -g 1 -r 6 -l CPU-CPU -d packed -b packed:1 \
  nvprof --analysis-metrics -o $HOME/MEMBERWORK/matmult-$matrix-1n.%q{OMPI_COMM_WORLD_RANK}.%h.%p.nvvp --profile-from-start off \
  ./sf_matmult -f ${matrix}.aij -mat_type aijcusparse -vec_type cuda -xy ${matrix}.xy -nskip 5 -niter 50 -use_gpu_aware_mpi 1
else
  echo "Test with $matrix with sfcuda without profiling\n"
  jsrun --smpiargs "-gpu" -n 6 -a 1 -c 2 -g 1 -r 6 -l CPU-CPU -d packed -b packed:1 \
   ./sf_matmult -f ${matrix}.aij -mat_type aijcusparse -vec_type cuda -xy ${matrix}.xy -nskip 5 -niter 50 -sf_backend cuda


  echo "Test with $matrix with sfkokkos without profiling\n"
  jsrun --smpiargs "-gpu" -n 6 -a 1 -c 2 -g 1 -r 6 -l CPU-CPU -d packed -b packed:1 \
   ./sf_matmult -f ${matrix}.aij -mat_type aijcusparse -vec_type cuda -xy ${matrix}.xy -nskip 5 -niter 50 -sf_backend kokkos
fi
done
