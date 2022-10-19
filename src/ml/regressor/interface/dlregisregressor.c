#include <petsc/private/regressorimpl.h>

static PetscBool PetscRegressorPackageInitialized = PETSC_FALSE;

/*@C
   PetscRegressorInitializePackage - Initialize PetscRegressor package

   Logically Collective

   Level: developer

.seealso: PetscRegressorFinalizePackage()
@*/
PetscErrorCode PetscRegressorInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscRegressorPackageInitialized) PetscFunctionReturn(0);
  PetscRegressorPackageInitialized = PETSC_TRUE;
  /* Register Class */
  ierr = PetscClassIdRegister("Regressor",&PETSCREGRESSOR_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscRegressorRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("PetscRegressorSetUp",PETSCREGRESSOR_CLASSID,&PetscRegressor_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscRegressorFit",PETSCREGRESSOR_CLASSID,&PetscRegressor_Fit);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscRegressorPredict",PETSCREGRESSOR_CLASSID,&PetscRegressor_Predict);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCREGRESSOR_CLASSID;
    ierr = PetscInfoProcessClass("petscregressor", 1, classids);CHKERRQ(ierr);
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscRegressorFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscRegressorFinalizePackage - Finalize PetscRegressor package; it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: PetscRegressorInitializePackage()
@*/
PetscErrorCode PetscRegressorFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscRegressorList);CHKERRQ(ierr);
  PetscRegressorPackageInitialized = PETSC_FALSE;
  PetscRegressorRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
