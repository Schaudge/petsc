
CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF  =
SOURCEH  = petsc.h petscsys.h petscmath.h petscmatlab.h petscbt.h \
        petscversion.h petscviewerhdf5.h \
        petscviewersaws.h petscwebclient.h petscblaslapack.h \
<<<<<<< HEAD
        petscsystypes.h petscdevicetypes.h petscdevice.h
=======
        petscsystypes.h petsclandau.h 
>>>>>>> Added RiemannSolver class to implement riemann solvers for the discretization of conservation/balance laws. This now allows for a physics independent method of specifying riemann solvers, allowing for resusablility and consistency. The simple local-lax fridrich solver was implemented and some basic tests were done. A WIP that will be moved to its own branch at some point
LIBBASE  = libpetscvec
DIRS     = petsc/finclude petsc/private
LOCDIR   = include/
MANSEC   = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
