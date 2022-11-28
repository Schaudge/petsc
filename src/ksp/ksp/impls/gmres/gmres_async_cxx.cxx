#include <petscmacros.h>

#if !PetscDefined(HAVE_CUDA)
  #include "gmres_async.hpp"
#endif
