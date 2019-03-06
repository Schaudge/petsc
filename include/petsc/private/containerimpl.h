
#if !defined(__CONTAINERIMPL)
#define __CONTAINERIMPL

#include <petsc/private/petscimpl.h>

struct _p_PetscContainer {
  PETSCHEADER(int);
  void           *ptr;
  PetscErrorCode (*userdestroy)(void*);
};

#endif
