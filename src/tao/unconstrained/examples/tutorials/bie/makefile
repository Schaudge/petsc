
# This directory contains access to matlab test problems for the derivative free
# least squares solvers.

ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = bieoptimizer.c
EXAMPLESF        = 
EXAMPLESCH       =
EXAMPLESFH       = 
EXAMPLESMATLAB   = initialize.m ObjectiveAndGradient.m 
LOCDIR		 = src/tao/unconstrained/examples/tutorials/bie
MANSEC		 =
DOCS		 =
DIRS		 =
CLEANFILES       = bieoptimizer

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


bieoptimizer: bieoptimizer.o chkopts
	-${CLINKER} -o bieoptimizer bieoptimizer.o ${PETSC_TAO_LIB}
	${RM} bieoptimizer.o
	

include ${PETSC_DIR}/lib/petsc/conf/test

