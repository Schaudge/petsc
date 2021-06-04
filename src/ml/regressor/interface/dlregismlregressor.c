#include <petsc/private/mlregressorimpl.h>

static PetscBool MLRegressorPackageInitialized = PETSC_FALSE;

static PetscClassId PETSCMLREGRESSOR_CLASSID;

/*@C
   MLRegressorInitializePackage - Initialize MLRegressor package

   Logically Collective

   Level: developer

.seealso: MLRegressorFinalizePackage()
@*/
PetscErrorCode MLRegressorInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MLRegressorPackageInitialized) PetscFunctionReturn(0);
  MLRegressorPackageInitialized = PETSC_TRUE;
  /* Register Class */
  ierr = PetscClassIdRegister("Regressor",&MLREGRESSOR_CLASSID);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  /* Register Constructors */
  ierr = MLRegressorRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  // TODO: Add events once I've defined some!
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCMLREGRESSOR_CLASSID;
    ierr = PetscInfoProcessClass("mlregressor", 1, classids);CHKERRQ(ierr);
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(MLRegressorFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MLRegressorFinalizePackage - Finalize MLRegressor package; it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: MLRegressorInitializePackage()
@*/
PetscErrorCode MLRegressorFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&MLRegressorList);CHKERRQ(ierr);
  MLRegressorPackageInitialized = PETSC_FALSE;
  MLRegressorRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
