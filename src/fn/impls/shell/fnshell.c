
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

PetscErrorCode PetscFnShellSetContext(PetscFn fn,void *ctx)
{
  PetscFn_Shell  *shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fn,PETSCFN_CLASSID,1,PETSCFNSHELL);
  shell = (PetscFn_Shell *) fn->data;
  shell->ctx = ctx;
  PetscFunctionReturn(0);
}

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
  case PETSCFNOP_JACOBIANMULT:
  case PETSCFNOP_JACOBIANMULTADJOINT:
  case PETSCFNOP_JACOBIANBUILD:
  case PETSCFNOP_JACOBIANBUILDADJOINT:
  case PETSCFNOP_HESSIANMULT:
  case PETSCFNOP_HESSIANMULTADJOINT:
  case PETSCFNOP_HESSIANBUILD:
  case PETSCFNOP_HESSIANBUILDADJOINT:
  case PETSCFNOP_HESSIANBUILDSWAP:
  case PETSCFNOP_SCALARGRADIENT:
  case PETSCFNOP_SCALARHESSIANMULT:
  case PETSCFNOP_SCALARHESSIANBUILD:
    break;
  case PETSCFNOP_SCALARAPPLY:
    fn->ops->scalarapply = (PetscErrorCode (*) (PetscFn,Vec,PetscReal *)) f;
    break;
  case PETSCFNOP_CREATESUBFNS:
    fn->ops->createsubfns = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,const IS[],const IS[],PetscFn *[])) f;
    break;
  case PETSCFNOP_DESTROYSUBFNS:
    fn->ops->destroysubfns = (PetscErrorCode (*) (PetscInt,PetscFn *[])) f;
    break;
  case PETSCFNOP_CREATESUBFN:
    fn->ops->createsubfn = (PetscErrorCode (*) (PetscFn,Vec,IS,IS,MatReuse,PetscFn *)) f;
    break;
  case PETSCFNOP_CREATEDERIVATIVEFN:
    fn->ops->createderivativefn = (PetscErrorCode (*) (PetscFn,PetscInt,PetscInt,PetscInt,const IS[],const Vec[],PetscFn *)) f;
    break;
  case PETSCFNOP_DESTROY:
    shell->destroy = (PetscErrorCode (*) (PetscFn)) f;
    break;
  case PETSCFNOP_VIEW:
    fn->ops->view = (PetscErrorCode (*) (PetscFn, PetscViewer)) f;
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
  case PETSCFNOP_SCALARDERIVATIVESCALAR:
    fn->ops->scalarderivativescalar = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,const IS[],const Vec[],PetscScalar *)) f;
    break;
  case PETSCFNOP_SCALARDERIVATIVEVEC:
    fn->ops->scalarderivativevec = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,const IS[],const Vec[],Vec)) f;
    break;
  case PETSCFNOP_SCALARDERIVATIVEMAT:
    fn->ops->scalarderivativemat = (PetscErrorCode (*) (PetscFn,Vec,PetscInt,const IS[],const Vec[],MatReuse,Mat*,Mat*)) f;
    break;
  }
  PetscFunctionReturn(0);
}

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
  case PETSCFNOP_JACOBIANMULT:
  case PETSCFNOP_JACOBIANMULTADJOINT:
  case PETSCFNOP_JACOBIANBUILD:
  case PETSCFNOP_JACOBIANBUILDADJOINT:
  case PETSCFNOP_HESSIANMULT:
  case PETSCFNOP_HESSIANMULTADJOINT:
  case PETSCFNOP_HESSIANBUILD:
  case PETSCFNOP_HESSIANBUILDADJOINT:
  case PETSCFNOP_HESSIANBUILDSWAP:
  case PETSCFNOP_SCALARGRADIENT:
  case PETSCFNOP_SCALARHESSIANMULT:
  case PETSCFNOP_SCALARHESSIANBUILD:
    *f = NULL;
    break;
  case PETSCFNOP_SCALARAPPLY:
    *f = (void (*)(void)) fn->ops->scalarapply;
    break;
  case PETSCFNOP_CREATESUBFNS:
    *f = (void (*)(void)) fn->ops->createsubfns;
    break;
  case PETSCFNOP_DESTROYSUBFNS:
    *f = (void (*)(void)) fn->ops->destroysubfns;
    break;
  case PETSCFNOP_CREATESUBFN:
    *f = (void (*)(void)) fn->ops->createsubfn;
    break;
  case PETSCFNOP_CREATEDERIVATIVEFN:
    *f = (void (*)(void)) fn->ops->createderivativefn;
    break;
  case PETSCFNOP_DESTROY:
    *f = (void (*)(void)) shell->destroy;
    break;
  case PETSCFNOP_VIEW:
    *f = (void (*)(void)) fn->ops->view;
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
  case PETSCFNOP_SCALARDERIVATIVESCALAR:
    *f = (void (*)(void)) fn->ops->scalarderivativescalar;
    break;
  case PETSCFNOP_SCALARDERIVATIVEVEC:
    *f = (void (*)(void)) fn->ops->scalarderivativevec;
    break;
  case PETSCFNOP_SCALARDERIVATIVEMAT:
    *f = (void (*)(void)) fn->ops->scalarderivativemat;
    break;
  }
  PetscFunctionReturn(0);
}

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
