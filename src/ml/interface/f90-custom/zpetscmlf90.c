#include <petscml.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define mldestroy_                  MLDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define mldestroy_                  mldestroy
#endif

PETSC_EXTERN void mldestroy_(ML *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = MLDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
