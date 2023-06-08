#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/prox/prox.h>


const char *const TaoPROXTypes[] = {"DEFAULT", "ADAPTIVE", "VM", "TaoPROXType", "TAO_PROX_", NULL};

/* Updates Moreau Regularization to the given objective and gradient.
 * For subsolverX, routine needs to be ComputeObjectiveAndGraidnet
 * Separate Objective and Gradient routines are not supported.  */
static PetscErrorCode AddMoreauRegObjGrad(Tao tao)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;

  // OBJ
  if (tao->ops->computeobjective) {
  
  }
  if (tao->ops->computegradient) {
  
  }
  //if above two happened, this shouldn't happen. bool trigger seterrq?
  if (tao->ops->computeobjectiveandgradient) {
  
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_PROX(Tao tao)
{
  TAO_PROX                     *proxP       = (TAO_PROX *)tao->data;
  TaoLineSearchConvergedReason  ls_status   = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                     step        = 1.0, f, gnorm, gnorm2, delta, gd, ginner, beta;

  PetscFunctionBegin;

  /*  Check convergence criteria */
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  /*  Set initial direction to -gradient */
  PetscCall(VecCopy(tao->gradient, tao->stepdirection));
  PetscCall(VecScale(tao->stepdirection, -1.0));
  gnorm2 = gnorm * gnorm;


  while (1) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);

    /*  Save the current gradient information */

    /*  Check for termination */
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_PROX(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  if (!proxP->X_old) PetscCall(VecDuplicate(tao->solution, &proxP->X_old));
  if (!proxP->G_old) PetscCall(VecDuplicate(tao->gradient, &proxP->G_old));

  PetscCall(AddMoreauRegObjGrad(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_PROX(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&proxP->X_old));
    PetscCall(VecDestroy(&proxP->G_old));
  }
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_PROX(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadBegin(PetscOptionsObject, "Nonlinear Conjugate Gradient method for unconstrained optimization");
  PetscCall(PetscOptionsEnum("-tao_prox_type", "TAOPROX update typess", "TaoPROXType", TaoPROXTypes, (PetscEnum)proxP->strategy, (PetscEnum *)&proxP->strategy, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_PROX(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_PROX   *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PROX Type: %s\n", TaoPROXTypes[proxP->strategy]));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TAOPROX -  Proximal algorithm.

   Solves proximal operator. User gives function, f(x), and this algorithm solves

   min_x f(x) + \rho/2 \|x-y \|_2^2.

   \rho is stepsize, which can be adaptive. (should i have user-given strategy for adaptiveness?? probably not...)
   
   Or, if the user preferrs, one can use variable-metric version, which is

   min_x f(x) + 1/2 \|x-y\|_M^2, where \|x\|_M^2 = x^T M x. VM will be diagonal. (future support for diag+ vv^T, a la becker et al?)

   Should default solve method be like CG? Gradient descent?

   "Built-in f(x)", see Beck ...

   Options Database Keys:

  Notes:
     PROX formulas are:
  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_PROX(Tao tao)
{
  TAO_PROX     *proxP;

  PetscFunctionBegin;
  PetscCall(PetscNew(&proxP));

  tao->ops->setup          = TaoSetUp_PROX;
  tao->ops->solve          = TaoSolve_PROX;
  tao->ops->view           = TaoView_PROX;
  tao->ops->setfromoptions = TaoSetFromOptions_PROX;
  tao->ops->destroy        = TaoDestroy_PROX;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  PetscCall(PetscNew(&proxP));
  tao->data                = (void *)proxP;
  proxP->eta               = 0.1;
  proxP->strategy          = TAO_PROX_DEFAULT;
  proxP->stepsize          = 1.;
  proxP->orig_objP         = NULL;
  proxP->orig_objgradP     = NULL;
  proxP->orig_gradP        = NULL;
  proxP->orig_hessP        = NULL;
  proxP->ops->orig_obj     = NULL;
  proxP->ops->orig_objgrad = NULL;
  proxP->ops->orig_grad    = NULL;
  proxP->ops->orig_hess    = NULL;

  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &proxP->subsolver));
  PetscCall(TaoSetOptionsPrefix(proxP->subsolver, "inner_prox_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)proxP->subsolver, (PetscObject)tao, 1));
  PetscCall(TaoSetType(proxP->subsolver, TAONM));
  
  PetscCall(PetscObjectCompose((PetscObject)proxP->subsolver, "TaoGetPROXParentTao_PROX", (PetscObject)tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXSetStepSize(Tao tao, PetscReal step)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->stepsize = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXGetStepSize(Tao tao, PetscReal *step)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *step = proxP->stepsize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXSetVM(Tao tao, Mat vm)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->vm = vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXGetVM(Tao tao, Mat *vm)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *vm = proxP->vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Custom set objective. I want user to be able to just simply
 * name the built-in functions, instead of their own??? */
PETSC_EXTERN PetscErrorCode TaoPROXSetObjective(Tao tao, TaoPROXFunc func_name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  tao->user_objP = NULL;
  tao->ops->computeobjective = ...;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TaoGetPROXParentTao - Gets pointer to parent `TAOPROX`, used by inner subsolver.

   Collective

   Input Parameter:
. tao - the `Tao` context

   Output Parameter:
. prox_tao - the parent `Tao` context

   Level: advanced

.seealso: `TAOPROX`
@*/
PetscErrorCode TaoGetPROXParentTao(Tao tao, Tao *prox_tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscObjectQuery((PetscObject)tao, "TaoGetPROXParentTao_ADMM", (PetscObject *)prox_tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

