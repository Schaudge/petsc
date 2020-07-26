#if !defined(_PETSCHYPRE_H)
#define _PETSCHYPRE_H

#include <petscsys.h>
#include <petscpkg_version.h>
#include <HYPRE_config.h>
#include <HYPRE_utilities.h>

/* from version 2.16 on, HYPRE_BigInt is 64 bit for 64bit installations
   and 32 bit for 32bit installations -> not the best name for a variable */
#if PETSC_PKG_HYPRE_VERSION_LT(2,16,0)
typedef PetscInt HYPRE_BigInt;
#endif

/*
  With scalar type == real, HYPRE_Complex == PetscScalar;
  With scalar type == complex,  HYPRE_Complex is double __complex__ while PetscScalar may be std::complex<double>
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHYPREScalarCast(PetscScalar a, HYPRE_Complex *b)
{
  PetscFunctionBegin;
#if defined(HYPRE_COMPLEX)
  ((PetscReal*)b)[0] = PetscRealPart(a);
  ((PetscReal*)b)[1] = PetscImaginaryPart(a);
#else
  *b = a;
#endif
  PetscFunctionReturn(0);
}

/* Protect with macros since hypre_HandleDefaultExecPolicy is exposed for all builds only from 2.19.0-devel (at some point) on
   replace with version check when released? */
#include <_hypre_utilities.h>
PETSC_STATIC_INLINE PetscErrorCode PetscHYPRESetExecPolicy(HYPRE_ExecutionPolicy policy)
{
  PetscFunctionBegin;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
  hypre_HandleDefaultExecPolicy(hypre_handle()) = policy;
#endif
  PetscFunctionReturn(0);
}

#endif
