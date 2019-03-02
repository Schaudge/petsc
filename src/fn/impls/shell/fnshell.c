
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

static PetscErrorCode PetscFnShellRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnShellRegister(PETSCSIN,         PetscFnShellCreate_Sin);CHKERRQ(ierr);
  ierr = PetscFnShellRegister(PETSCNORMSQUARED, PetscFnShellCreate_Normsquared);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnShellInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
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

PetscErrorCode PetscFnShellCreate(MPI_Comm comm, PetscFnShellType shelltype,PetscInt m, PetscInt M, PetscInt n, PetscInt N, void *ctx, PetscFn *fn_p)
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
    fn->ops->createvecs = (PetscErrorCode (*) (PetscFn,Vec*,Vec*)) f;
    break;
  case PETSCFNOP_CREATEMATS:
    fn->ops->createmats = (PetscErrorCode (*) (PetscFn,PetscFnOperation,Mat*,Mat*)) f;
    break;
  case PETSCFNOP_APPLY:
    fn->ops->apply = (PetscErrorCode (*) (PetscFn,Vec,Vec)) f;
    break;
  case PETSCFNOP_JACOBIANMULT:
    fn->ops->jacobianmult = (PetscErrorCode (*) (PetscFn,Vec,Vec,Vec)) f;
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    fn->ops->jacobianmultadjoint = (PetscErrorCode (*) (PetscFn,Vec,Vec,Vec)) f;
    break;
  case PETSCFNOP_JACOBIANBUILD:
    fn->ops->jacobianbuild = (PetscErrorCode (*) (PetscFn,Vec,Mat,Mat)) f;
    break;
  case PETSCFNOP_JACOBIANBUILDADJOINT:
    fn->ops->jacobianbuildadjoint = (PetscErrorCode (*) (PetscFn,Vec,Mat,Mat)) f;
    break;
  case PETSCFNOP_HESSIANMULT:
    fn->ops->hessianmult = (PetscErrorCode (*) (PetscFn,Vec,Vec,Vec,Vec)) f;
    break;
  case PETSCFNOP_HESSIANMULTADJOINT:
    fn->ops->hessianmultadjoint = (PetscErrorCode (*) (PetscFn,Vec,Vec,Vec,Vec)) f;
    break;
  case PETSCFNOP_HESSIANBUILD:
    fn->ops->hessianbuild = (PetscErrorCode (*) (PetscFn,Vec,Vec,Mat,Mat)) f;
    break;
  case PETSCFNOP_HESSIANBUILDADJOINT:
    fn->ops->hessianbuildadjoint = (PetscErrorCode (*) (PetscFn,Vec,Vec,Mat,Mat)) f;
    break;
  case PETSCFNOP_HESSIANBUILDSWAP:
    fn->ops->hessianbuildswap = (PetscErrorCode (*) (PetscFn,Vec,Vec,Mat,Mat)) f;
    break;
  case PETSCFNOP_SCALARAPPLY:
    fn->ops->scalarapply = (PetscErrorCode (*) (PetscFn,Vec,PetscReal *)) f;
    break;
  case PETSCFNOP_SCALARGRADIENT:
    fn->ops->scalargradient = (PetscErrorCode (*) (PetscFn,Vec,Vec)) f;
    break;
  case PETSCFNOP_SCALARHESSIANMULT:
    fn->ops->scalarhessianmult = (PetscErrorCode (*) (PetscFn,Vec,Vec,Vec)) f;
    break;
  case PETSCFNOP_SCALARHESSIANBUILD:
    fn->ops->scalarhessianbuild = (PetscErrorCode (*) (PetscFn,Vec,Mat,Mat)) f;
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
    fn->ops->createderivativefn = (PetscErrorCode (*) (PetscFn,PetscFnOperation,PetscInt,const Vec[],PetscFn *)) f;
    break;
  case PETSCFNOP_DESTROY:
    shell->destroy = (PetscErrorCode (*) (PetscFn)) f;
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
  case PETSCFNOP_CREATEMATS:
    *f = (void (*)(void)) fn->ops->createmats;
    break;
  case PETSCFNOP_APPLY:
    *f = (void (*)(void)) fn->ops->apply;
    break;
  case PETSCFNOP_JACOBIANMULT:
    *f = (void (*)(void)) fn->ops->jacobianmult;
    break;
  case PETSCFNOP_JACOBIANMULTADJOINT:
    *f = (void (*)(void)) fn->ops->jacobianmultadjoint;
    break;
  case PETSCFNOP_JACOBIANBUILD:
    *f = (void (*)(void)) fn->ops->jacobianbuild;
    break;
  case PETSCFNOP_JACOBIANBUILDADJOINT:
    *f = (void (*)(void)) fn->ops->jacobianbuildadjoint;
    break;
  case PETSCFNOP_HESSIANMULT:
    *f = (void (*)(void)) fn->ops->hessianmult;
    break;
  case PETSCFNOP_HESSIANMULTADJOINT:
    *f = (void (*)(void)) fn->ops->hessianmultadjoint;
    break;
  case PETSCFNOP_HESSIANBUILD:
    *f = (void (*)(void)) fn->ops->hessianbuild;
    break;
  case PETSCFNOP_HESSIANBUILDADJOINT:
    *f = (void (*)(void)) fn->ops->hessianbuildadjoint;
    break;
  case PETSCFNOP_HESSIANBUILDSWAP:
    *f = (void (*)(void)) fn->ops->hessianbuildswap;
    break;
  case PETSCFNOP_SCALARAPPLY:
    *f = (void (*)(void)) fn->ops->scalarapply;
    break;
  case PETSCFNOP_SCALARGRADIENT:
    *f = (void (*)(void)) fn->ops->scalargradient;
    break;
  case PETSCFNOP_SCALARHESSIANMULT:
    *f = (void (*)(void)) fn->ops->scalarhessianmult;
    break;
  case PETSCFNOP_SCALARHESSIANBUILD:
    *f = (void (*)(void)) fn->ops->scalarhessianbuild;
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
