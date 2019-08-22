#include <petsc/private/taomeritimpl.h>

static PetscErrorCode TaoMeritGetValue_Objective(TaoMerit merit, PetscReal alpha, PetscReal *fval)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(merit->Xinit, merit->Xtrial);CHKERRQ(ierr);
  ierr = VecAXPY(merit->Xtrial, alpha, merit->step);CHKERRQ(ierr);
  ierr = TaoMeritComputeObjective(merit, merit->Xtrial, fval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoMeritGetDirDeriv_Objective(TaoMerit merit, PetscReal alpha, PetscReal *gts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(merit->Xinit, merit->Xtrial);CHKERRQ(ierr);
  ierr = VecAXPY(merit->Xtrial, alpha, merit->step);CHKERRQ(ierr);
  ierr = TaoMeritComputeGradient(merit, merit->Xtrial, merit->Gtrial);CHKERRQ(ierr);
  ierr = VecDot(merit->step, merit->Gtrial, gts);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoMeritGetValueAndDirDeriv_Objective(TaoMerit merit, PetscReal alpha, PetscReal *fval, PetscReal *gts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(merit->Xinit, merit->Xtrial);CHKERRQ(ierr);
  ierr = VecAXPY(merit->Xtrial, alpha, merit->step);CHKERRQ(ierr);
  ierr = TaoMeritComputeObjectiveAndGradient(merit, merit->Xtrial, fval, merit->Gtrial);CHKERRQ(ierr);
  ierr = VecDot(merit->step, merit->Gtrial, gts);
  PetscFunctionReturn(0);
}

/*MC
   TAOMERITOBJECTIVE - Basic merit function for the objective function

   Level: developer

.seealso: TaoMeritCreate(), TaoMeritSetType()

.keywords: Tao, merit
M*/
PETSC_EXTERN PetscErrorCode TaoMeritCreate_Objective(TaoMerit merit)
{
  PetscFunctionBegin;
  merit->ops->getvalue = TaoMeritGetValue_Objective;
  merit->ops->getdirderiv = TaoMeritGetDirDeriv_Objective;
  merit->ops->getvalueanddirderiv = TaoMeritGetValueAndDirDeriv;
  PetscFunctionReturn(0);

}


