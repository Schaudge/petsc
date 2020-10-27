#!/usr/bin/env python
#
# Currently Loaded Modules:
#  1) hsi/5.0.2.p5   2) xalt/1.2.0   3) lsf-tools/2.0   4) darshan-runtime/3.1.7   5) DefApps   6) cmake/3.18.2   7) cuda/10.1.243   8) gcc/6.4.0   9) spectrum-mpi/10.3.1.2-20200121  10) netlib-lapack/3.8.0  11) forge/20.0.1
#

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-fc=0',
    '--COPTFLAGS=-g -fPIC',
    '--CXXOPTFLAGS=-g -fPIC ',
    '--FOPTFLAGS=-g -fPIC ',
    '--CUDAOPTFLAGS=-g -Xcompiler -rdynamic -lineinfo'
    '--CUDAFLAGS=-arch=sm_70',
    '--with-ssl=0',
    '--with-batch=0',
    '--with-cxx=mpicxx',
    '--with-mpiexec=jsrun -g1 --smpiargs "-gpu"',
    '--with-cuda=1',
    '--with-cudac=nvcc',
#    '--download-p4est=1',
#    '--download-zlib',
#    '--download-hdf5=1',
    '--download-metis',
#    '--download-superlu_dist',
#    '--download-superlu_dist-commit=HEAD',
#    '--download-hypre-configure-arguments=HYPRE_CUDA_SM=70',
    #'--with-hwloc=0',
    '--download-parmetis',
    #'--download-hypre',
    '--download-triangle',
    #'--download-amgx',
    #'--download-fblaslapack',
    '--with-blaslapack-lib=-L' + os.environ['OLCF_NETLIB_LAPACK_ROOT'] + '/lib64 -lblas -llapack',
    #'--download-openblas',
    '--with-cc=mpicc',
    #'--with-fc=mpif90',
    '--with-shared-libraries=1',
    #  '--known-mpi-shared-libraries=1',
    '--with-x=0',
    '--with-64-bit-indices=0',
    '--with-debugging=0',
    '--download-kokkos',
    '--with-ctable=1',
    '--with-make-np=8',
    '--download-kokkos-kernels',
    '--with-kokkos-cuda-arch=VOLTA70',
    '--with-kokkos-kernels-tpl=0',
    'PETSC_ARCH=arch-summit-opt-gnu-kokkos-notpl-cuda',
#    '--prefix=/gpfs/alpine/world-shared/geo127/petsc/opt-gcc-int64-cuda-omp',
  ]
  configure.petsc_configure(configure_options)
  
