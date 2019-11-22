#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC   = 
SOURCECU  = pmiscuda.cu
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/coarsen/impls/mis/cuda/
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
