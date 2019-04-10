
#include <petsc/private/fnimpl.h> /*I "petscfn.h" I*/
#include <petscfn.h>

PetscBool PetscFnShellRegisterAllCalled = PETSC_FALSE;
PetscBool PetscFnShellPackageInitialized = PETSC_FALSE;

PetscFunctionList PetscFnShellList = 0;

static PetscErrorCode PetscFnShellFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscFnShellList);CHKERRQ(ierr);
  PetscFnShellPackageInitialized = PETSC_FALSE;
  PetscFnShellRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellInitializePackage(void);

PetscErrorCode PetscFnShellRegister(const char name[],PetscErrorCode (*shell)(PetscFn))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscFnShellList,name,shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFnShellCreate_Sin(PetscFn);
PETSC_EXTERN PetscErrorCode PetscFnShellCreate_Normsquared(PetscFn);
PETSC_EXTERN PetscErrorCode PetscFnShellCreate_Mat(PetscFn);

static PetscErrorCode PetscFnShellRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellRegister(PETSCFNSIN,         PetscFnShellCreate_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellRegister(PETSCFNNORMSQUARED, PetscFnShellCreate_Normsquared);CHKERRQ(ierr);
  ierr = PetscFnShellRegister(PETSCFNMAT,         PetscFnShellCreate_Mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscFnShellPackageInitialized) PetscFunctionReturn(0);
  PetscFnShellPackageInitialized = PETSC_TRUE;
  /* Register Constructors */
  ierr = PetscFnShellRegisterAll();CHKERRQ(ierr);

  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscFnShellFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnShellCreate(MPI_Comm comm, PetscFnShellType shelltype, PetscInt m, PetscInt n, PetscInt M, PetscInt N, void *ctx, PetscFn *fn_p)
{
  PetscFn        fn;
  PetscErrorCode (*r) (PetscFn);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnCreate(comm, &fn);CHKERRQ(ierr);
  ierr = PetscFnSetSizes(fn, m, n, M, N);CHKERRQ(ierr);
  ierr = PetscFnSetType(fn, PETSCFNSHELL);CHKERRQ(ierr);
  ierr = PetscFnShellSetContext(fn, ctx);CHKERRQ(ierr);
  if (shelltype) {
    ierr =  PetscFunctionListFind(PetscFnShellList,shelltype,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscFnShellType given: %s",shelltype);
    ierr = (*r)(fn);CHKERRQ(ierr);
  }
  *fn_p = fn;
  PetscFunctionReturn(0);
}

typedef struct {
  PetscErrorCode (*destroy)(PetscFn);
  void *ctx;
} PetscFn_Shell;

/*@
    PetscFnShellSetContext - sets the context for a shell function

   Logically Collective on PetscFn

    Input Parameters:
+   fn - the shell matrix
-   ctx - the context

   Level: advanced

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

.seealso: PETSCFNSHELL, PetscFnShellGetContext(), PetscFnShellSetOperation()
@*/
PetscErrorCode PetscFnShellSetContext(PetscFn fn,void *ctx)
{
  PetscFn_Shell  *shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fn,PETSCFN_CLASSID,1,PETSCFNSHELL);
  shell = (PetscFn_Shell *) fn->data;
  shell->ctx = ctx;
  PetscFunctionReturn(0);
}

/*@
    PetscFnShellGetContext - Returns the user-provided context associated with a shell function.

    Not Collective

    Input Parameter:
.   fn - the function

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

.seealso: PETSCFNSHELL, PetscFnShellSetOperation(), PetscFnShellSetContext()
@*/
PetscErrorCode PetscFnShellGetContext(PetscFn fn,void *ctx)
{
  PetscFn_Shell  *shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fn,PETSCFN_CLASSID,1,PETSCFNSHELL);
  shell = (PetscFn_Shell *) fn->data;
  *((void **) ctx) = shell->ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDestroy_Shell(PetscFn fn)
{
  PetscFn_Shell  *shell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  shell = (PetscFn_Shell *) fn->data;
  if (shell->destroy) {ierr = (*(shell->destroy)) (fn);CHKERRQ(ierr);}
  ierr = PetscFree(fn->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscFnShellSetOperation - Allows user to set a PetscFn operation for a shell function.

   Logically Collective on PetscFn

    Input Parameters:
+   fn - the shell function
.   op - the name of the operation
-   f - the function that provides the operation.

   Level: advanced

    Usage:
$      extern PetscErrorCode userapply(PetscFn,Vec,Vec);
$      ierr = PetscFnCreate(comm,&fn);
$      ierr = PetscFnSetSizes(fn,m,n,M,N);
$      ierr = PetscFnShellSetContext(fn,ctx);
$      ierr = PetscFnShellSetOperation(A,PETSCFNOP_APPLY,(void(*)(void))userapply);

    Notes:
    See the file include/petscfn.h for a complete list of function
    operations, which all have the form PETSCFNOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., PetscFnApply() -> PETSCFNOP_APPLY).

    All user-provided functions (except for PETSCFNOP_DESTROY) should have the same calling
    sequence as the usual function interface routines, since they
    are intended to be accessed via the usual PetscFn interface
    routines, e.g.,
$       PetscFnApply(Mat,Vec,Vec) -> userapply(Mat,Vec,Vec)

    In particular each function MUST return an error code of 0 on success and
    nonzero on failure.

    Within each user-defined routine, the user should call
    PetscFnShellGetContext() to obtain the user-defined context that was
    set by PetscFnShellSetContext().

.keywords: PetscFn, shell, set, operation

.seealso: PETSCFNSHELL, PetscFnShellGetContext(), PetscFnShellGetOperation(), PetscFnShellSetContext(),
@*/
PetscErrorCode PetscFnShellSetOperation(PetscFn fn, PetscFnOperation op, void (*f)(void))
{
  PetscFn_Shell  *shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fn,PETSCFN_CLASSID,1,PETSCFNSHELL);
  shell = (PetscFn_Shell *) fn->data;

  switch (op) {
  case PETSCFNOP_CREATEVECS:
    fn->ops->createvecs = (PetscErrorCode (*) (PetscFn,IS,Vec*,IS,Vec*)) f;
    break;
  case PETSCFNOP_APPLY:
    fn->ops->apply = (PetscErrorCode (*) (PetscFn,Vec,Vec)) f;
    break;
  case PETSCFNOP_DERIVATIVESCALAR:
    fn->ops->derivativescalar = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],PetscScalar *)) f;
    break;
  case PETSCFNOP_DERIVATIVEVEC:
    fn->ops->derivativevec = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],Vec)) f;
    break;
  case PETSCFNOP_DERIVATIVEMAT:
    fn->ops->derivativemat = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],MatReuse,Mat*,Mat*)) f;
    break;
  case PETSCFNOP_DERIVATIVEFN:
    fn->ops->derivativefn = (PetscErrorCode (*) (PetscFn,PetscInt,PetscInt,PetscInt,const IS[],const Vec[],PetscFn *)) f;
    break;
  case PETSCFNOP_SCALARAPPLY:
    fn->ops->scalarapply = (PetscErrorCode (*) (PetscFn,Vec,PetscReal *)) f;
    break;
  case PETSCFNOP_SCALARDERIVATIVESCALAR:
    fn->ops->scalarderivativescalar = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,const IS[],const Vec[],PetscScalar *)) f;
    break;
  case PETSCFNOP_SCALARDERIVATIVEVEC:
    fn->ops->scalarderivativevec = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,const IS[],const Vec[],Vec)) f;
    break;
  case PETSCFNOP_SCALARDERIVATIVEMAT:
    fn->ops->scalarderivativemat = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,const IS[],const Vec[],MatReuse,Mat*,Mat*)) f;
    break;
  case PETSCFNOP_SCALARDERIVATIVEFN:
    fn->ops->scalarderivativefn = (PetscErrorCode (*) (PetscFn,PetscInt,PetscInt,const IS[],const Vec[],PetscFn *)) f;
    break;
  case PETSCFNOP_DESTROY:
    shell->destroy = (PetscErrorCode (*) (PetscFn)) f;
    break;
  case PETSCFNOP_VIEW:
    fn->ops->view = (PetscErrorCode (*) (PetscFn, PetscViewer)) f;
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscFnShellGetOperation - Gets a PetscFn operation for a shell function.

    Not Collective

    Input Parameters:
+   fn - the shell function
-   op - the name of the operation

    Output Parameter:
.   f - the function that provides the operation.

    Level: advanced

    Notes:
    See the file include/petscfn.h for a complete list of matrix
    operations, which all have the form PETSCFNOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., PetscFnApply() -> PETSCFNOP_APPLY).

    All user-provided functions (except for PETSCFNOP_DESTROY) should have the same calling
    sequence as the usual function interface routines, since they
    are intended to be accessed via the usual PetscFn interface
    routines, e.g.,
$       PetscFnApply(Mat,Vec,Vec) -> userapply(Mat,Vec,Vec)

    In particular each function MUST return an error code of 0 on success and
    nonzero on failure.

    Within each user-defined routine, the user should call
    PetscFnShellGetContext() to obtain the user-defined context that was
    set by PetscFnShellSetContext().

.keywords: matrix, shell, set, operation

.seealso: PetscFnShellGetContext(), PetscFnShellSetOperation(), PetscFnShellSetContext()
@*/
PetscErrorCode PetscFnShellGetOperation(PetscFn fn, PetscFnOperation op, void (**f)(void))
{
  PetscFn_Shell  *shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fn,PETSCFN_CLASSID,1,PETSCFNSHELL);
  shell = (PetscFn_Shell *) fn->data;

  switch (op) {
  case PETSCFNOP_CREATEVECS:
    *f = (void (*)(void)) fn->ops->createvecs;
    break;
  case PETSCFNOP_APPLY:
    *f = (void (*)(void)) fn->ops->apply;
    break;
  case PETSCFNOP_DERIVATIVESCALAR:
    *f = (void (*)(void)) fn->ops->derivativescalar;
    break;
  case PETSCFNOP_DERIVATIVEVEC:
    *f = (void (*)(void)) fn->ops->derivativevec;
    break;
  case PETSCFNOP_DERIVATIVEMAT:
    *f = (void (*)(void)) fn->ops->derivativemat;
    break;
  case PETSCFNOP_DERIVATIVEFN:
    *f = (void (*)(void)) fn->ops->derivativefn;
    break;
  case PETSCFNOP_SCALARAPPLY:
    *f = (void (*)(void)) fn->ops->scalarapply;
    break;
  case PETSCFNOP_SCALARDERIVATIVESCALAR:
    *f = (void (*)(void)) fn->ops->scalarderivativescalar;
    break;
  case PETSCFNOP_SCALARDERIVATIVEVEC:
    *f = (void (*)(void)) fn->ops->scalarderivativevec;
    break;
  case PETSCFNOP_SCALARDERIVATIVEMAT:
    *f = (void (*)(void)) fn->ops->scalarderivativemat;
    break;
  case PETSCFNOP_SCALARDERIVATIVEFN:
    *f = (void (*)(void)) fn->ops->scalarderivativefn;
    break;
  case PETSCFNOP_DESTROY:
    *f = (void (*)(void)) shell->destroy;
    break;
  case PETSCFNOP_VIEW:
    *f = (void (*)(void)) fn->ops->view;
    break;
  }
  PetscFunctionReturn(0);
}

/*MC
  SNESSHELL - a user provided function

   Level: advanced

.seealso: PetscFnCreate(),
          PetscFn,
          PetscFnSetType(),
          PetscFnType (for list of available types),
          PetscFnShellSetContext(),
          PetscFnShellGetContext(),
          PetscFnShellSetOperation()
          PetscFnShellGetOperation()
M*/
PetscErrorCode PetscFnCreate_Shell(PetscFn fn)
{
  PetscFn_Shell  *shell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellInitializePackage();CHKERRQ(ierr);
  ierr = PetscNewLog(fn, &shell);CHKERRQ(ierr);
  fn->data = shell;
  fn->ops->destroy = PetscFnDestroy_Shell;
  ierr = PetscObjectChangeTypeName((PetscObject)fn,PETSCFNSHELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
