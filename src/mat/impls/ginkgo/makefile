-include ../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_GINKGO'

ALL: lib

FFLAGS   =
SOURCECXX= matginkgo.cxx
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/ginkgo/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
