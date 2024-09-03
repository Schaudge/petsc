#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

static PetscErrorCode TaoTermObjective_Halfl2squared(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  PetscScalar v;
  Vec         diff;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(x, &diff));
  PetscCall(VecCopy(x, diff));
  PetscCall(VecAXPY(diff, -1.0, params));
  PetscCall(VecDot(diff, diff, &v));
  *value = 2 * PetscRealPart(v);
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_Halfl2squared(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  PetscScalar v;

  PetscFunctionBegin;
  PetscCall(VecWAXPY(g, -1.0, params, x));
  PetscCall(VecDot(g, g, &v));
  *value = 2 * PetscRealPart(v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_Halfl2squared(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscFunctionBegin;
  PetscCall(VecWAXPY(g, -1.0, params, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_Halfl2squared(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatShift(H, 1.0));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMHALFL2SQUARED - A `TaoTerm` that computes $\tfrac{1}{2}\|x - p\|_2^2$, for solution $x$ and parameters $p$.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetProximalMap()`,
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_L2squared(TaoTerm term)
{
  PetscFunctionBegin;
  term->ops->objective            = TaoTermObjective_Halfl2squared;
  term->ops->gradient             = TaoTermGradient_Halfl2squared;
  term->ops->objectiveandgradient = TaoTermObjectiveAndGradient_Halfl2squared;
  term->ops->hessian              = TaoTermHessian_Halfl2squared;
  PetscFunctionReturn(PETSC_SUCCESS);
}
