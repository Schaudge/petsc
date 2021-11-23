-include ../../../../../../../petscdir.mk
ALL: lib

CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/network/tutorials/dgnet/CouplingConditions
EXAMPLESC       = ex2.c ex3.c
EXAMPLESF       =
EXAMPLESCH      = 
MANSEC          = DM
SUBMANSEC       = DMNetwork
SOURCES_DGNET   = $(wildcard ../*.c)
OBJECTS_DGNET   = $(patsubst %.c,%.o,$(SOURCES_DGNET))

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex2 : $(OBJECTS_DGNET)
ex3 : $(OBJECTS_DGNET)