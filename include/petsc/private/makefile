
CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF  =
SOURCEH = bagimpl.h characteristicimpl.h dmdaimpl.h dmforestimpl.h      \
 dmimpl.h dmlabelimpl.h dmmbimpl.h dmnetworkimpl.h dmpatchimpl.h        \
 dmpleximpl.h dmswarmimpl.h drawimpl.h dtimpl.h f90impl.h               \
 fortranimpl.h isimpl.h kspimpl.h linesearchimpl.h logimpl.h            \
 matimpl.h matorderimpl.h pcgamgimpl.h pcimpl.h pcmgimpl.h              \
 petscdsimpl.h petscfeimpl.h petscfptimpl.h petscfvimpl.h petscimpl.h   \
 sfimpl.h snesimpl.h taoimpl.h taolinesearchimpl.h tsimpl.h vecimpl.h   \
 tshistoryimpl.h                                                        \
 hashtable.h hashmap.h hashset.h                                        \
<<<<<<< HEAD
 viewerimpl.h viewerhdf5impl.h partitionerimpl.h
=======
 viewerimpl.h viewerhdf5impl.h partitionerimpl.h                        \
 petscriemannsolverimpl.h
>>>>>>> Added RiemannSolver class to implement riemann solvers for the discretization of conservation/balance laws. This now allows for a physics independent method of specifying riemann solvers, allowing for resusablility and consistency. The simple local-lax fridrich solver was implemented and some basic tests were done. A WIP that will be moved to its own branch at some point
LIBBASE  = libpetscvec
DIRS     =
LOCDIR   = include/petsc/private/
MANSEC   = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
