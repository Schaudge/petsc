
#include <petscvec.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/veccupmimpl.h>

PETSC_EXTERN PetscErrorCode VecCreate_MPI(Vec, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_Standard(Vec, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_Shared(Vec, PetscDeviceContext);
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
PETSC_EXTERN PetscErrorCode VecCreate_Node(Vec, PetscDeviceContext);
#endif
#if defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_MPIViennaCL(Vec, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_ViennaCL(Vec, PetscDeviceContext);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode VecCreate_SeqKokkos(Vec, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_MPIKokkos(Vec, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_Kokkos(Vec, PetscDeviceContext);
#endif

/*@C
  VecRegisterAll - Registers all of the vector components in the Vec package.

  Not Collective, Synchronous

  Level: advanced

.seealso: `VecRegister()`, `VecRegisterDestroy()`, `VecRegister()`
@*/
PetscErrorCode VecRegisterAll(void) {
  PetscFunctionBegin;
  if (VecRegisterAllCalled) PetscFunctionReturn(0);
  VecRegisterAllCalled = PETSC_TRUE;

  PetscCall(VecRegister(VECSEQ, VecCreate_Seq));
  PetscCall(VecRegister(VECMPI, VecCreate_MPI));
  PetscCall(VecRegister(VECSTANDARD, VecCreate_Standard));
  PetscCall(VecRegister(VECSHARED, VecCreate_Shared));
#if defined PETSC_HAVE_VIENNACL
  PetscCall(VecRegister(VECSEQVIENNACL, VecCreate_SeqViennaCL));
  PetscCall(VecRegister(VECMPIVIENNACL, VecCreate_MPIViennaCL));
  PetscCall(VecRegister(VECVIENNACL, VecCreate_ViennaCL));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(VecRegister(VECSEQCUDA, VecCreate_SeqCUDA));
  PetscCall(VecRegister(VECMPICUDA, VecCreate_MPICUDA));
  PetscCall(VecRegister(VECCUDA, VecCreate_CUDA));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(VecRegister(VECSEQKOKKOS, VecCreate_SeqKokkos));
  PetscCall(VecRegister(VECMPIKOKKOS, VecCreate_MPIKokkos));
  PetscCall(VecRegister(VECKOKKOS, VecCreate_Kokkos));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(VecRegister(VECSEQHIP, VecCreate_SeqHIP));
  PetscCall(VecRegister(VECMPIHIP, VecCreate_MPIHIP));
  PetscCall(VecRegister(VECHIP, VecCreate_HIP));
#endif
  PetscFunctionReturn(0);
}
