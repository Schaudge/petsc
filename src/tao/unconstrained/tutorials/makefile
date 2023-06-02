-include ../../../../petscdir.mk

MANSEC		 = Tao
CLEANFILES       = eptorsion1 eptorsion2 minsurf2 rosenbrock1 eptorsion2f rosenbrock1f rosenbrock2 rosenbrock3

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

rosenbrock1objs := rosenbrock1.o
ifneq (&(CUDAC),)
  rosenbrock1objs += rosenbrock1cu.o
endif
rosenbrock1 : ${rosenbrock1objs}
