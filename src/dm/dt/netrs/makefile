-include ../../../../petscdir.mk
ALL: lib

SOURCEC   = fluxfun.c
SOURCEH   = ../../../../include/petsc/private/netrsimpl.h ../../../../include/petscnetrs.h fluxfun.h
DIRS      = interface impls tests
LOCDIR    = src/dm/dt/netrs
MANSEC    = DM
SUBMANSEC = NRS

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

