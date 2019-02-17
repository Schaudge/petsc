
#include <petsc/private/fnimpl.h> /*I "petscfn.h" I*/
#include <petscfn.h>

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
  case PETSCFNOP_CREATE_VECS:
    fn->ops->createvecs = (PetscErrorCode (*) (PetscFn,Vec*,Vec*)) f;
    break;
  case PETSCFNOP_CREATE_MATS:
    fn->ops->createmats = (PetscErrorCode (*) (PetscFn,Mat*,Mat*,Mat*,Mat*,Mat*,Mat*)) f;
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
  case PETSCFNOP_JACOBIANCREATE:
    fn->ops->jacobiancreate = (PetscErrorCode (*) (PetscFn,Vec,Mat,Mat)) f;
    break;
  case PETSCFNOP_JACOBIANCREATEADJOINT:
    fn->ops->jacobiancreateadjoint = (PetscErrorCode (*) (PetscFn,Vec,Mat,Mat)) f;
    break;
  case PETSCFNOP_HESSIANMULT:
    fn->ops->hessianmult = (PetscErrorCode (*) (PetscFn,Vec,Vec,Vec,Vec)) f;
    break;
  case PETSCFNOP_HESSIANCREATE:
    fn->ops->hessiancreate = (PetscErrorCode (*) (PetscFn,Vec,Vec,Mat,Mat)) f;
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
  case PETSCFNOP_SCALARHESSIANCREATE:
    fn->ops->scalarhessiancreate = (PetscErrorCode (*) (PetscFn,Vec,Mat,Mat)) f;
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
  case PETSCFNOP_CREATE_VECS:
    *f = (void (*)(void)) fn->ops->createvecs;
    break;
  case PETSCFNOP_CREATE_MATS:
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
  case PETSCFNOP_JACOBIANCREATE:
    *f = (void (*)(void)) fn->ops->jacobiancreate;
    break;
  case PETSCFNOP_JACOBIANCREATEADJOINT:
    *f = (void (*)(void)) fn->ops->jacobiancreateadjoint;
    break;
  case PETSCFNOP_HESSIANMULT:
    *f = (void (*)(void)) fn->ops->hessianmult;
    break;
  case PETSCFNOP_HESSIANCREATE:
    *f = (void (*)(void)) fn->ops->hessiancreate;
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
  case PETSCFNOP_SCALARHESSIANCREATE:
    *f = (void (*)(void)) fn->ops->scalarhessiancreate;
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
  ierr = PetscNewLog(fn, &shell);CHKERRQ(ierr);
  fn->data = shell;
  fn->ops->destroy = PetscFnDestroy_Shell;
  ierr = PetscObjectChangeTypeName((PetscObject)fn,PETSCFNSHELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
