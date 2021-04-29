#!/bin/bash
### Begin BSUB Options
#BSUB -P CSC314
#BSUB -J o-pingpong-2n
#BSUB -W 00:20
#BSUB -nnodes 2
#BSUB -alloc_flags "smt4 gpumps"
### End BSUB Options and begin shell commands

date
module list
echo PAMI_CUDA_AWARE_THRESH=$PAMI_CUDA_AWARE_THRESH

printf "\n\nIntra-socket Ping-pong test\n"

printf "OSU Host memory\n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -l GPU-GPU -d packed -b packed:7 /ccs/home/jczhang/osu-micro-benchmarks-5.7/mpi/pt2pt/osu_latency H H

printf "OSU Device memory\n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -l GPU-GPU -d packed -b packed:7 /ccs/home/jczhang/osu-micro-benchmarks-5.7/mpi/pt2pt/osu_latency D D

printf "SF Host memory\n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -l GPU-GPU -d packed -b packed:7 ./sf_pingpong -mtype host

printf "CUDA memory + -use_gpu_aware_mpi 1 \n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -l GPU-GPU -d packed -b packed:7 ./sf_pingpong -mtype cuda -use_gpu_aware_mpi 1 -use_nvshmem 0

printf "CUDA memory + -use_nvshmem 1 \n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -l GPU-GPU -d packed -b packed:7 ./sf_pingpong -mtype cuda -use_nvshmem 1


printf "\n\nInter-socket Ping-pong test\n"

printf "OSU Host memory\n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -d packed -b packed:21 /ccs/home/jczhang/osu-micro-benchmarks-5.7/mpi/pt2pt/osu_latency H H

printf "OSU Device memory\n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -d packed -b packed:21 /ccs/home/jczhang/osu-micro-benchmarks-5.7/mpi/pt2pt/osu_latency D D

printf "Host memory\n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -d packed -b packed:21 ./sf_pingpong -mtype host

printf "CUDA memory + -use_gpu_aware_mpi 1 \n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -d packed -b packed:21 ./sf_pingpong -mtype cuda -use_gpu_aware_mpi 1 -use_nvshmem 0

printf "CUDA memory + -use_nvshmem 1 \n"
jsrun --smpiargs "-gpu" -n 1 -a 2 -c 42 -g 2 -r 1 -d packed -b packed:21 ./sf_pingpong -mtype cuda -use_nvshmem 1


printf "\n\nInter-node Ping-pong test\n"

printf "OSU Host memory\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 42 -g 1 -r 1 -d packed -b packed:7 /ccs/home/jczhang/osu-micro-benchmarks-5.7/mpi/pt2pt/osu_latency H H

printf "OSU Device memory\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 42 -g 1 -r 1 -d packed -b packed:7 /ccs/home/jczhang/osu-micro-benchmarks-5.7/mpi/pt2pt/osu_latency D D

printf "Host memory\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 42 -g 1 -r 1 -d packed -b packed:7 ./sf_pingpong -mtype host

printf "CUDA memory + -use_gpu_aware_mpi 1 \n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 42 -g 1 -r 1 -d packed -b packed:7 ./sf_pingpong -mtype cuda -use_gpu_aware_mpi 1 -use_nvshmem 0

printf "CUDA memory + -use_nvshmem 1 \n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 42 -g 1 -r 1 -d packed -b packed:7 ./sf_pingpong -mtype cuda -use_nvshmem 1
