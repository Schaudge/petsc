#define TAOPD_DLL
#include <petsc/private/taopdimpl.h>

PETSC_EXTERN PetscErrorCode TaoPDCreate_L1_Private(TaoPD);
PETSC_EXTERN PetscErrorCode TaoPDCreate_L2_Private(TaoPD);
PETSC_EXTERN PetscErrorCode TaoPDCreate_Simplex_Private(TaoPD);
static PetscBool            TaoPDPackageInitialized = PETSC_FALSE;

/*@C
  TaoPDFinalizePackage - This function destroys everything in the `TaoPD` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: `Tao`, `TaoPD`
@*/
PetscErrorCode TaoPDFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoPDList));
  TaoPDPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDInitializePackage - This function registers the problem description
  routine in `Tao`.  When using shared or static libraries, this function is called from the
  first entry to `TaoCreate()`; when using dynamic, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

.seealso: `Tao`, `TaoPD`, `TaoPDCreate()`
@*/
PetscErrorCode TaoPDInitializePackage(void)
{
  PetscFunctionBegin;
  if (TaoPDPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TaoPDPackageInitialized = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscClassIdRegister("TaoPD", &TAOPD_CLASSID));
  PetscCall(TaoPDRegister("l2", TaoPDCreate_L2_Private));
  PetscCall(TaoPDRegister("l1", TaoPDCreate_L1_Private));
  PetscCall(TaoPDRegister("simplex", TaoPDCreate_Simplex_Private));
  PetscCall(PetscLogEventRegister("TaoPDEval", TAOPD_CLASSID, &TAOPD_Eval));
  PetscCall(PetscLogEventRegister("TaoPDApply", TAOPD_CLASSID, &TAOPD_Apply));
#endif
  PetscCall(PetscRegisterFinalize(TaoPDFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
