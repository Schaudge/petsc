#include "sindy_impl.h"

static PetscBool SINDyPackageInitialized     = PETSC_FALSE;
static PetscBool VariablePackageInitialized  = PETSC_FALSE;
static PetscBool SparseRegPackageInitialized = PETSC_FALSE;


/*@C
  SINDyFinalizePackage - This function destroys everything in the SINDy package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode SINDyFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SINDyPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
      SINDyInitializePackage - This function initializes everything in the SINDy
  package. It is called on the first call to SINDyBasisCreate() when using
  shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode SINDyInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SINDyPackageInitialized) PetscFunctionReturn(0);
  SINDyPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("SINDy",    &SINDY_CLASSID);CHKERRQ(ierr);

  /* Register Constructors */
  ierr = PetscLogEventRegister("SINDyBasisCreate",SINDY_CLASSID,&SINDy_BasisCreate);CHKERRQ(ierr);

  /* Register Events */
  ierr = PetscLogEventRegister("SINDyFindSparseCoefficients", SINDY_CLASSID,&SINDy_FindSparseCoefficients);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SINDyBasisPrint",             SINDY_CLASSID,&SINDy_BasisPrint);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SINDyBasisAddVariables",      SINDY_CLASSID,&SINDy_BasisAddVariables);CHKERRQ(ierr);


  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = SINDY_CLASSID;
    ierr = PetscInfoProcessClass("sindy",     1, &classids[0]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("sindy",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(SINDY_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(SINDyFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VariableFinalizePackage - This function destroys everything in the Variable
  package. It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode VariableFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VariablePackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
      VariableInitializePackage - This function initializes everything in the Variable
  package. It is called on the first call to VariableCreate() when using
  shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode VariableInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (VariablePackageInitialized) PetscFunctionReturn(0);
  VariablePackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Variable", &VARIABLE_CLASSID);CHKERRQ(ierr);

  /* Register Events */
  ierr = PetscLogEventRegister("VariablePrint",               VARIABLE_CLASSID,&Variable_Print);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VariableDifferentiateSpatial",VARIABLE_CLASSID,&Variable_DifferentiateSpatial);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VariableExtractDataByDim",    VARIABLE_CLASSID,&Variable_ExtractDataByDim);CHKERRQ(ierr);

  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = VARIABLE_CLASSID;
    ierr = PetscInfoProcessClass("variable",  1, &classids[0]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("variable",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(VARIABLE_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(VariableFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  SparseRegFinalizePackage - This function destroys everything in the SparseReg
  package. It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode SparseRegFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SparseRegPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
      SparseRegInitializePackage - This function initializes everything in the SparseReg
  package. It is called on the first call to SparseRegCreate() when using
  shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode SparseRegInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SparseRegPackageInitialized) PetscFunctionReturn(0);
  SparseRegPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("SparseReg",&SPARSEREG_CLASSID);CHKERRQ(ierr);

  /* Register Events */
  ierr = PetscLogEventRegister("SparseRegSTLSQ",              SPARSEREG_CLASSID,&SparseReg_STLSQ);CHKERRQ(ierr);

  /* Process Info */
  {
    PetscClassId  classids[0];

    classids[0] = SPARSEREG_CLASSID;
    ierr = PetscInfoProcessClass("sparsereg", 1, &classids[0]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("sparsereg",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(SPARSEREG_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(SparseRegFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
