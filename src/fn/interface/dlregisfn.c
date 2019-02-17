
#include <petsc/private/fnimpl.h>

static PetscBool PetscFnPackageInitialized = PETSC_FALSE;
/*@C
  PetscFnFinalizePackage - This function destroys everything in the Petsc interface to the PetscFn package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscFnFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscFnList);CHKERRQ(ierr);
  PetscFnPackageInitialized = PETSC_FALSE;
  PetscFnRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscFnInitializePackage - This function initializes everything in the PetscFn package. It is called
  from PetscDLLibraryRegister_petscfn() when using dynamic libraries, and on the first call to PetscFnCreate()
  when using shared or static libraries.

  Level: developer

.keywords: PetscFn, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscFnInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscFnPackageInitialized) PetscFunctionReturn(0);
  PetscFnPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Nonlinear Function",&PETSCFN_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscFnRegisterAll();CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("fn",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscInfoDeactivateClass(PETSCFN_CLASSID);CHKERRQ(ierr);}
  }

  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("fn",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSCFN_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscFnFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the PetscFn methods that are in the basic PETSc Matrix library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscfn(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
