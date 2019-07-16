
# This directory contains access to matlab test problems for the derivative free
# least squares solvers.

ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = bieoptimizer.c bieoptimizerMirror.c
EXAMPLESF        = 
EXAMPLESCH       =
EXAMPLESFH       = 
EXAMPLESMATLAB   = initialize.m ObjectiveAndGradient.m ObjectiveAndGradientMirror.m
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
bieoptimizerMirror: bieoptimizerMirror.o chkopts
	-${CLINKER} -o bieoptimizerMirror bieoptimizerMirror.o ${PETSC_TAO_LIB}
	${RM} bieoptimizerMirror.o
	

include ${PETSC_DIR}/lib/petsc/conf/test

