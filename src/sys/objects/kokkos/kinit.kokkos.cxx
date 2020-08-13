#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <Kokkos_Core.hpp>

PetscErrorCode PetscKokkosInitialize(void)
{
  Kokkos::InitArguments args;
  int                   devId;

  PetscFunctionBegin;
#if defined(KOKKOS_ENABLE_CUDA)
  cudaGetDevice(&devId);
#elif defined(KOKKOS_ENABLE_HIP)
  hipGetDevice(&devId);
#endif
  args.device_id = devId;
  Kokkos::initialize(args);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscKokkosInitialized(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  *isInitialized = Kokkos::is_initialized() ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscKokkosFinalize(void)
{
  PetscFunctionBegin;
  Kokkos::finalize();
  PetscFunctionReturn(0);
}