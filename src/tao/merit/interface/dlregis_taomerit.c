#define TAOMERIT_DLL
#include <petsc/private/taomeritimpl.h>


PETSC_EXTERN PetscErrorCode TaoMeritCreate_Objective(Tao, TaoMerit);
PETSC_EXTERN PetscErrorCode TaoMeritCreate_Lagrangian(Tao, TaoMerit);
PETSC_EXTERN PetscErrorCode TaoMeritCreate_AugLag(Tao, TaoMerit);
PETSC_EXTERN PetscErrorCode TaoMeritCreate_LogBarrier(Tao, TaoMerit);
static PetscBool TaoMeritPackageInitialized = PETSC_FALSE;

/*@C
  TaoMeritFinalizePackage - This function destroys everything in the PETSc/TAO
  interface to the TaoMerit package. It is called from PetscFinalize().

  Level: developer
@*/
PetscErrorCode TaoMeritFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&TaoMeritList);CHKERRQ(ierr);
  TaoMeritPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritInitializePackage - This function registers the merit function types 
  in TAO.  When using shared or static libraries, this function is called from the
  first entry to TaoCreate(); when using dynamic, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

.seealso: TaoMeritCreate()
@*/
PetscErrorCode TaoMeritInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TaoMeritPackageInitialized) PetscFunctionReturn(0);
  TaoMeritPackageInitialized=PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscClassIdRegister("TaoMerit",&TAOMERIT_CLASSID);CHKERRQ(ierr);
  ierr = TaoLineSearchRegister("objective",TaoMeritCreate_Objective);CHKERRQ(ierr);
  ierr = TaoLineSearchRegister("lagrangian",TaoMeritCreate_Lagrangian);CHKERRQ(ierr);
  ierr = TaoLineSearchRegister("auglag",TaoMeritCreate_AugLag);CHKERRQ(ierr);
  ierr = TaoLineSearchRegister("logbarrier",TaoMeritCreate_LogBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoMeritEvaluate", TAOMERIT_CLASSID,&TAOMERIT_Eval);CHKERRQ(ierr);
#endif
  ierr = PetscRegisterFinalize(TaoMeritFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



