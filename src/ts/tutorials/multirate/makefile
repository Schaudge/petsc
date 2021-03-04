-include ../../../../petscdir.mk
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/tutorials/multirate/
<<<<<<< HEAD
<<<<<<< HEAD
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex9.c
=======
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
OBJECTS_FV      = finitevolume1d.o
DIRS            = fvnet
<<<<<<< HEAD

SOURCES_FVNET = $(wildcard ./fvnet/*.c)
<<<<<<< HEAD
OBJECTS_FVNET = $(patsubst %.c,%.o,$(SOURCES_FVNET)) 
=======
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex9.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
<<<<<<< HEAD
OBJECTS_FV      = finitevolume1d.o limiters.o fvnet.o
>>>>>>> Added ex9, network multirate example, fixed various bugs. Code now runs however is not yet yeild the correct results
=======
OBJECTS_FV      = finitevolume1d.o
DIRS            = fvnet

SOURCES_FVNET = $(wildcard ./fvnet/*.c)
OBJECTS_FVNET = $(patsubst %.c,%.o,$(SOURCES_FVNET)) 
>>>>>>> Fixed memory leak. Reorganized the file structure.
=======
OBJECTS_FVNET = $(patsubst %.c,%.o,$(SOURCES_FVNET))
>>>>>>> rm white spaces;
=======
SOURCES_FVNET   = $(wildcard ./fvnet/*.c)
OBJECTS_FVNET   = $(patsubst %.c,%.o,$(SOURCES_FVNET))
CLEANFILES      = ${OBJECTS_FVNET}
>>>>>>> Modified make file to improve dependency on fvnet.h WIP

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex4: ex4.o $(OBJECTS_FV)
ex5: ex5.o $(OBJECTS_FV)
ex4: ex4.o $(OBJECTS_FV)
ex6: ex6.o $(OBJECTS_FV)
ex8: ex8.o $(OBJECTS_FV)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
ex9: ex9.o $(OBJECTS_FVNET)
=======
ex9: ex9.o $(OBJECTS_FV)
>>>>>>> Added ex9, network multirate example, fixed various bugs. Code now runs however is not yet yeild the correct results
=======
ex9: ex9.o $(OBJECTS_FVNET)
>>>>>>> Fixed memory leak. Reorganized the file structure.

=======
ex9.o:  ./fvnet/fvnet.h ./fvnet/limiters.h
ex9: ex9.o $(OBJECTS_FVNET)
>>>>>>> Modified make file to improve dependency on fvnet.h WIP
include ${PETSC_DIR}/lib/petsc/conf/test

