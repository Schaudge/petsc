-include ../../../../../../../petscdir.mk
ALL: lib

CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/network/tutorials/dgnet/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c
EXAMPLESF       =
EXAMPLESCH      = 
MANSEC          = DM
SUBMANSEC       = DMNetwork
SOURCES_DGNET   = $(wildcard ../*.c)
OBJECTS_DGNET   = $(patsubst %.c,%.o,$(SOURCES_DGNET))
SOURCES_WASH = $(wildcard ../hydronetwork-2021/src/river/*.c ../hydronetwork-2021/src/*.c)
OBJECTS_RIVER = $(patsubst %.c,%.o,$(SOURCES_WASH))

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex1 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex2 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex3 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex4 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex5 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex6 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex7 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex8 : $(OBJECTS_DGNET) $(OBJECTS_RIVER)

ex9:  $(OBJECTS_DGNET) $(OBJECTS_RIVER)
ex10 : $(OBJECTS_DGNET)  $(OBJECTS_RIVER)
ex11 : $(OBJECTS_DGNET)  $(OBJECTS_RIVER)

.PHONY: all
all : ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10

VALUE = 10

clean ::
	rm ${OBJECTS_RIVER} ${OBJECTS_DGNET}