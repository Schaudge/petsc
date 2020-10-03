#!/bin/bash
### Begin BSUB Options
#BSUB -P CSC314
#BSUB -J o-sf-one-node
#BSUB -W 00:20
#BSUB -nnodes 1
#BSUB -alloc_flags "smt4 gpumps"
### End BSUB Options and begin shell commands

date
module list

printf "\n\nInter-GPU intra-socket new Ping-pong test with -sf_use_default_stream\n"
printf "https://jsrunvisualizer.olcf.ornl.gov/?s4f0o01n2c7g1r11d1b27l0=\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 7 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./sf_newpingpong  -mtype device

printf "\n\nInter-GPU intra-socket Unpack test with -sf_use_default_stream\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 7 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./sf_unpack  -mtype device

printf "\n\nInter-GPU intra-socket Scatter test with -sf_use_default_stream\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 7 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./sf_scatter  -mtype device -nprof 0

printf "\n\nInter-GPU inter-socket new Ping-pong test with -sf_use_default_stream\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 21 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./sf_newpingpong -mtype device

printf "\n\nInter-GPU inter-socket Unpack test with -sf_use_default_stream\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 21 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./sf_unpack -mtype device

printf "\n\nInter-GPU inter-socket Scatter test with -sf_use_default_stream\n"
jsrun --smpiargs "-gpu" -n 2 -a 1 -c 21 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./sf_scatter -mtype device -nprof 0
