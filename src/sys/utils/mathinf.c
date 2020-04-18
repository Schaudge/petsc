#define PETSC_SKIP_COMPLEX
#include <petscsys.h>
/*@C
      PetscIsNormalReal - Returns PETSC_TRUE if the input value satisfies isnormal()

    Input Parameter:
.     a - the PetscReal Value

     Notes:
    uses the C99 standard fpclassify() on systems where they exist, or isnormal().
      Always returns true with __float128
      Otherwises always returns true

     Level: beginner
@*/
#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
PetscBool PetscIsNormalReal(PetscReal a)
{
  return PETSC_TRUE;
}
#elif defined(PETSC_USE_C99)
PetscBool PetscIsNormalReal(PetscReal a)
{
  return fpclassify(a) == FP_NORMAL ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE_ISNORMAL)
PetscBool PetscIsNormalReal(PetscReal a)
{
  return isnormal(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsNormalReal(PetscReal a)
{
  return PETSC_TRUE;
}
#endif

/*@C
      PetscIsInfReal - Returns whether the input is an infinity value.

    Input Parameter:
.     a - the floating point number

     Notes:
    uses the C99 standard fpclassify() on systems where it exists, or isinf().
      Uses isinfq() with __float128
      Otherwises uses (a && a/2 == a), note that some optimizing compiles compile
      out this form, thus removing the check.

     Level: beginner
@*/
#if defined(PETSC_USE_REAL___FLOAT128)
PetscBool PetscIsInfReal(PetscReal a)
{
  return isinfq(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_USE_C99)
PetscBool PetscIsInfReal(PetscReal a)
{
  return fpclassify(a) == FP_INFINITE ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE_ISINF)
PetscBool PetscIsInfReal(PetscReal a)
{
  return isinf(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE__FINITE)
#if defined(PETSC_HAVE_FLOAT_H)
#include <float.h>  /* Microsoft Windows defines _finite() in float.h */
#endif
#if defined(PETSC_HAVE_IEEEFP_H)
#include <ieeefp.h>  /* Solaris prototypes these here */
#endif
PetscBool PetscIsInfReal(PetscReal a)
{
  return !_finite(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsInfReal(PetscReal a)
{
  return (a && a/2 == a) ? PETSC_TRUE : PETSC_FALSE;
}
#endif

/*@C
      PetscIsNanReal - Returns whether the input is a Not-a-Number (NaN) value.

    Input Parameter:
.     a - the floating point number

     Notes:
    uses the C99 standard fpclassify() on systems where it exists, or isnan().
      Uses isnanq() with __float128
      Otherwises uses (a != a), note that some optimizing compiles compile
      out this form, thus removing the check.

     Level: beginner
@*/
#if defined(PETSC_USE_REAL___FLOAT128)
PetscBool PetscIsNanReal(PetscReal a)
{
  return isnanq(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_USE_C99)
PetscBool PetscIsNanReal(PetscReal a)
{
  return fpclassify(a) == FP_NAN ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE_ISNAN)
PetscBool PetscIsNanReal(PetscReal a)
{
  return isnan(a) ? PETSC_TRUE : PETSC_FALSE;
}
#elif defined(PETSC_HAVE__ISNAN)
#if defined(PETSC_HAVE_FLOAT_H)
#include <float.h>  /* Microsoft Windows defines _isnan() in float.h */
#endif
#if defined(PETSC_HAVE_IEEEFP_H)
#include <ieeefp.h>  /* Solaris prototypes these here */
#endif
PetscBool PetscIsNanReal(PetscReal a)
{
  return _isnan(a) ? PETSC_TRUE : PETSC_FALSE;
}
#else
PetscBool PetscIsNanReal(PetscReal a)
{
  return (a != a) ? PETSC_TRUE : PETSC_FALSE;
}
#endif
