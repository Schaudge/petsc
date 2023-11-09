#include <petsc/private/mldrimpl.h>

static PetscBool MLDRPackageInitialized = PETSC_FALSE;

static PetscClassId PETSCMLDR_CLASSID;

/*@C
   MLDRInitializePackage - Initialize MLDR package

   Logically Collective

   Level: developer

.seealso: MLDRFinalizePackage()
@*/
PetscErrorCode MLDRInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MLDRPackageInitialized) PetscFunctionReturn(0);
  MLDRPackageInitialized = PETSC_TRUE;
  /* Register Class */
  ierr = PetscClassIdRegister("Dimension Reduction",&MLDR_CLASSID);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  /* Register Constructors */
  ierr = MLDRRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  // TODO: Add events once I've defined some!
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCMLDR_CLASSID;
    ierr = PetscInfoProcessClass("mldr", 1, classids);CHKERRQ(ierr);
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(MLDRFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MLDRFinalizePackage - Finalize MLDR package; it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: MLDRInitializePackage()
@*/
PetscErrorCode MLDRFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&MLDRList);CHKERRQ(ierr);
  MLDRPackageInitialized = PETSC_FALSE;
  MLDRRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
