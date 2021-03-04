-include ../../../../petscdir.mk
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/tutorials/multirate/fvnet/
EXAMPLESC       = fvfunctions.c fvnet.c fvnetmprk.c fvnetts.c limiters.c
EXAMPLESF       =
EXAMPLESCH      = limiters.h fvnet.h
MANSEC          = TS

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


fvfunctions.o : fvnet.h 
fvnetts.o     : fvnet.h
fvnetmprk.o   : fvnet.h