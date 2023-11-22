#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/regressorimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscregressordestroy_  PETSCREGRESSORDESTROY
  #define petscregressorsettype_  PETSCREGRESSORSETTYPE
  #define petscregressorview_     PETSCREGRESSORVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscregressordestroy_  petscregressordestroy
  #define petscregressorsettype_  petscregressorsettype
  #define petscregressorview_     petscregressorview
#endif


PETSC_EXTERN void petscregressordestroy_(PetscRegressor *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = PetscRegressorDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}

PETSC_EXTERN void petscregressorsettype_(PetscRegressor *reg, char *type_name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name, len, t);
  *ierr = PetscRegressorSetType(*reg, t);
  if (*ierr) return;
  FREECHAR(type_name, t);
}

PETSC_EXTERN void petscregressorview_(PetscRegressor *reg, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscRegressorView(*reg, v);
}
