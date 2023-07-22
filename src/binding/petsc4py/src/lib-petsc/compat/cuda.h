#ifndef PETSC4PY_COMPAT_CUDA_H
#define PETSC4PY_COMPAT_CUDA_H

#if !defined(PETSC_HAVE_CUDA)

#define PetscCUDAError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires CUDA",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode MatDenseCUDAGetArrayRead(PETSC_UNUSED Mat A, PETSC_UNUSED const PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDARestoreArrayRead(PETSC_UNUSED Mat A, PETSC_UNUSED const PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDAGetArrayWrite(PETSC_UNUSED Mat A, PETSC_UNUSED PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDARestoreArrayWrite(PETSC_UNUSED Mat A, PETSC_UNUSED PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDAGetArray(PETSC_UNUSED Mat A, PETSC_UNUSED PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDARestoreArray(PETSC_UNUSED Mat A, PETSC_UNUSED PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatCreateDenseCUDA(PETSC_UNUSED MPI_Comm comm, PETSC_UNUSED PetscInt m, PETSC_UNUSED PetscInt n, PETSC_UNUSED PetscInt M, PETSC_UNUSED PetscInt N, PETSC_UNUSED PetscScalar gpuarray[], PETSC_UNUSED Mat *A) {PetscCUDAError;}

#undef PetscCUDAError

#endif

#endif/*PETSC4PY_COMPAT_CUDA_H*/
