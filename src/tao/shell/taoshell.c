#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoShell Tao_Shell;

struct _n_TaoShell
{
  PetscErrorCode (*solve)(Tao, void *);
  void *ctx;
};

static PetscErrorCode TaoSolve_Shell(Tao tao)
{
  Tao_Shell                    *shell = (Tao_Shell*)tao->data;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = (*(shell->solve)) (tao, shell->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_Shell(Tao tao)
{
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoShellSetSolve(Tao tao, PetscErrorCode (*solve) (Tao, void *), void *ctx)
{
  Tao_Shell                    *shell = (Tao_Shell*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  shell->solve = solve;
  shell->ctx = ctx;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TaoCreate_Shell(Tao tao)
{
  Tao_Shell      *shell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->solve = TaoSolve_Shell;
  tao->ops->destroy = TaoDestroy_Shell;

  ierr = PetscNewLog(tao,&shell);CHKERRQ(ierr);
  tao->data = (void*)shell;
  PetscFunctionReturn(0);
}

