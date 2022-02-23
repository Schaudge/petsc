#requiresdefine 'PETSC_HAVE_CUPM'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   =
SOURCEF   =
SOURCEH   = vecseqcupm.hpp
LIBBASE   = libpetscvec
MANSEC    = Vec
LOCDIR    = src/vec/vec/impls/seq/cupm/
DIRS      = cuda hip

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
