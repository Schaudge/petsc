-include ../../../../../../petscdir.mk
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/tutorials/multirate/dgnet/CouplingConditions
EXAMPLESC       = ex1.c ex2.c ex3.c
EXAMPLESF       =
EXAMPLESCH      = 
MANSEC          = TS
SOURCES_DGNET   = $(wildcard ../*.c)
OBJECTS_DGNET   = $(patsubst %.c,%.o,$(SOURCES_DGNET))

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex1 : $(OBJECTS_DGNET)
ex2 : $(OBJECTS_DGNET)
ex3 : $(OBJECTS_DGNET)