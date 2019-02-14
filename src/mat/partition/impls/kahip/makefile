#
ALL: lib

#requirespackage 'PETSC_HAVE_KAHIP'

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${KAHIP_INCLUDE}
SOURCEC   = kahip.c
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/partition/impls/kahip/
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
