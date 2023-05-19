#ifndef PETSC_KOKKOS_HPP
#define PETSC_KOKKOS_HPP

#include <petscconf.h>

#if defined(petsccomplexlib)
  #error "Error: You must include petsc_kokkos.hpp before other petsc headers in this C++ file to use petsc complex with Kokkos"
#endif

#define PETSC_DESIRE_KOKKOS_COMPLEX 1 /* To control the definition of petsccomplexlib in petscsystypes.h */

#include <Kokkos_Core.hpp>

/* SUBMANSEC = Sys */

extern Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr;

/*MC
  PetscGetKokkosExecutionSpace - Return the Kokkos execution space that petsc is using

  Level: beginner

M*/
inline Kokkos::DefaultExecutionSpace &PetscGetKokkosExecutionSpace(void)
{
  return *PetscKokkosExecutionSpacePtr;
}

template <class IndexType>
inline decltype(auto) PetscKokkosRangePolicy(IndexType n)
{
  return Kokkos::RangePolicy<IndexType>(PetscGetKokkosExecutionSpace(), 0, n);
}

#endif
