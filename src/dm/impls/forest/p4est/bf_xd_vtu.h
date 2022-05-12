#if !defined(PETSCDMBF_VTU_XD_H)
#define PETSCDMBF_VTU_XD_H

#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

#if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL___FP16)
/* output in float if single or half precision in memory */
static const char precision[] = "Float32";
typedef float PetscVTUReal;
#define MPIU_VTUREAL MPI_FLOAT
#elif defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
/* output in double if double or quad precision in memory */
static const char precision[] = "Float64";
typedef double PetscVTUReal;
#define MPIU_VTUREAL MPI_DOUBLE
#else
static const char precision[] = "UnknownPrecision";
typedef PetscReal PetscVTUReal;
#define MPIU_VTUREAL MPIU_REAL
#endif

#define P4EST_VTK_CELL_TYPE      8      /* VTK_PIXEL */

#endif