#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext dctx)
{
  static const Petsc::CUPMContextHip  contextHip;
  PetscDeviceContext_(HIP)           *dci;
  PetscErrorCode                      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = static_cast<decltype(dctx->data)>(dci);
  ierr = PetscMemcpy(dctx->ops,&contextHip.ops,sizeof(contextHip.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Management of ROCBLAS and ROCSOLVER handles

 Unlike CUDA, rocsolver is just for dense matrices so there is
 no distinguishing being dense and sparse.  
 */

PetscErrorCode PetscHIPBLASGetHandle(rocblas_handle *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_HIP);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Because signatures are the same, have to deprecate to enable the function
 * overloading 
PetscErrorCode PetscHIPSOLVERGetHandle(rocblas_handle *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_HIP);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 * */
