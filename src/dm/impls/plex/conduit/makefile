#requirespackage  'PETSC_HAVE_CONDUIT'

ALL: lib

CXXPPFLAGS = $(CONDUIT_INCLUDE)
SOURCECXX  = plexconduit.cxx
LIBBASE    = libpetscdm
MANSEC     = DM
SUBMANSEC  = DMPLEX
LOCDIR     = src/dm/impls/plex/conduit/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
