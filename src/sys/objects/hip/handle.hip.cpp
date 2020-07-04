/*
 Management of HIPBLAS and HIPSOLVER handles
 */

#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <petschipblas.h>

static hipblasHandle_t     hipblasv2handle   = NULL;
static hipsolverDnHandle_t hipsolverdnhandle = NULL;

/*
   Destroys the HIPBLAS handle.
   This function is intended and registered for PetscFinalize - do not call manually!
 */
static PetscErrorCode PetscHIPBLASDestroyHandle()
{
  hipblasStatus_t cberr;

  PetscFunctionBegin;
  if (hipblasv2handle) {
    cberr          = hipblasDestroy(hipblasv2handle);CHKERRHIPBLAS(cberr);
    hipblasv2handle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

/*
    Initializing the hipBLAS handle can take 1/2 a second therefore
    initialize in PetscInitialize() before being timing so it does
    not distort the -log_view information
*/
PetscErrorCode PetscHIPBLASInitializeHandle(void)
{
  PetscErrorCode ierr;
  hipblasStatus_t cberr;

  PetscFunctionBegin;
  if (!hipblasv2handle) {
    cberr = hipblasCreate(&hipblasv2handle);CHKERRHIPBLAS(cberr);
    /* Make sure that the handle will be destroyed properly */
    ierr = PetscRegisterFinalize(PetscHIPBLASDestroyHandle);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  if (!hipblasv2handle) {ierr = PetscHIPBLASInitializeHandle();CHKERRQ(ierr);}
  *handle = hipblasv2handle;
  PetscFunctionReturn(0);
}

/* hipsolver */
static PetscErrorCode PetscHIPSOLVERDnDestroyHandle()
{
  hipsolverStatus_t  cerr;

  PetscFunctionBegin;
  if (hipsolverdnhandle) {
    cerr             = hipsolverDnDestroy(hipsolverdnhandle);CHKERRHIPSOLVER(cerr);
    hipsolverdnhandle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHIPSOLVERDnInitializeHandle(void)
{
  PetscErrorCode    ierr;
  hipsolverStatus_t  cerr;

  PetscFunctionBegin;
  if (!hipsolverdnhandle) {
    cerr = hipsolverDnCreate(&hipsolverdnhandle);CHKERRHIPSOLVER(cerr);
    ierr = PetscRegisterFinalize(PetscHIPSOLVERDnDestroyHandle);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHIPSOLVERDnGetHandle(hipsolverDnHandle_t *handle)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  if (!hipsolverdnhandle) {ierr = PetscHIPSOLVERDnInitializeHandle();CHKERRQ(ierr);}
  *handle = hipsolverdnhandle;
  PetscFunctionReturn(0);
}
