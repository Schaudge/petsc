-include ../../../../petscdir.mk
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/tutorials/multirate/dgnet/
EXAMPLESC       = ex1.c
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
ex4 : $(OBJECTS_DGNET)
ex5 : $(OBJECTS_DGNET)
ex6 : $(OBJECTS_DGNET)
ex7 : $(OBJECTS_DGNET)
ex8 : $(OBJECTS_DGNET)