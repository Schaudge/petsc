#requirespackage  'PETSC_HAVE_CONDUIT'

ALL: lib

CXXPPFLAGS = ${CONDUIT_INCLUDE}
SOURCECXX  = conduit.cxx
MANSEC     = Sys
SUBMANSEC  = Viewer
LIBBASE    = libpetscsys
LOCDIR     = src/sys/classes/viewer/impls/conduit/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
