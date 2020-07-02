-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_MAGMA'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = densemagma.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/dense/seq/magma/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
