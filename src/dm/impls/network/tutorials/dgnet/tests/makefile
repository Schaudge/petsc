-include ../../../../../../../petscdir.mk
ALL: lib

CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/network/tutorials/dgnet/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c
EXAMPLESF       =
EXAMPLESCH      = 
MANSEC          = DM
SUBMANSEC       = DMNetwork
SOURCES_DGNET   = $(wildcard ../*.c)
OBJECTS_DGNET   = $(patsubst %.c,%.o,$(SOURCES_DGNET))

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex1 : $(OBJECTS_DGNET)
ex2 : $(OBJECTS_DGNET)
ex3 : $(OBJECTS_DGNET)
ex4 : $(OBJECTS_DGNET)
ex5 : $(OBJECTS_DGNET)
ex6 : $(OBJECTS_DGNET)
ex7 : $(OBJECTS_DGNET)
ex8 : $(OBJECTS_DGNET)

.PHONY: all 
all : ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9