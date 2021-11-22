-include ../../../../../../petscdir.mk
ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEC  = physics.c dgnet.c dgfunctions.c
SOURCEF  =
SOURCEH  = dgnet.h physics.h 
DIRS     = tests CouplingConditions
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMNetwork
LOCDIR   = src/dm/impls/network/tutorials/dgnet/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
