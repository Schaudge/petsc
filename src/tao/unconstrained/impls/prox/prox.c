#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petscmath.h>
#include <../src/tao/unconstrained/impls/prox/prox.h>

//User means solving it via user specified types, e.g., TAOCG. 
//Metric would internally add MR.
//For User, TAOTYPE may not be TAOProx.
//TODO do I even need bregman here???
const char *const TaoProxStrategies[] = {"STRATEGY_DEFAULT", "STRATEGY_ADAPTIVE", "STRATEGY_VM", "TaoProxStrategy", "TAO_Prox_", NULL};

static PetscErrorCode TaoSolve_Prox(Tao tao)
{
  TAO_PROX     *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (proxP->y == NULL) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Need to set y vector for TAOProx first.");
  }

  /* Set metric */
  //TODO should strategy only for scalar stepsize? if its VM, should it be just left to METRIC?
  //PetscCall(TaoGetMetricType(tao, &metric_type));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);

    /* Solve */
    PetscCall(TaoApplyProximalMap(tao, proxP->stepsize,  proxP->y, tao->solution));

    /* Update stepsize / VM  TODO */
    /*  Check for termination */
    tao->niter++;
    //TODO unclear how to track this....
//    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
//    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
//    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_Prox(Tao tao)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  if (!proxP->X_old) PetscCall(VecDuplicate(tao->solution, &proxP->X_old));
  if (!proxP->G_old) PetscCall(VecDuplicate(tao->gradient, &proxP->G_old));
  if (!proxP->workvec1) PetscCall(VecDuplicate(tao->solution, &proxP->workvec1));
  if (!proxP->workvec1) PetscCall(VecDuplicate(tao->solution, &proxP->workvec1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_Prox(Tao tao)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&proxP->X_old));
    PetscCall(VecDestroy(&proxP->G_old));
    PetscCall(VecDestroy(&proxP->workvec1));
    PetscCall(VecDestroy(&proxP->y));
  }
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoProxMSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoProxMGetType_C", NULL));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_Prox(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Proximal algorithm optimization");
  PetscCall(PetscOptionsEnum("-tao_prox_strategy", "TAOProx update strategies", "TaoProxStrategy", TaoProxStrategies, (PetscEnum)proxP->strategy, (PetscEnum *)&proxP->strategy, NULL));
  /* Not supporting options to change prox type, as it doesn't makes sense */
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_Prox(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Prox Strategy: %s\n", TaoProxStrategies[proxP->strategy]));
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

   Issue: prox_f(y) = inf_x f(x) \rho/2 \|x-y\|_2^2.

   But TAO is essentially, min_x T(x). 

   If we are thinking in non=iterative, but just direct ones, (1-iter), this is easy.

   If we want to solve via other methods, it gets tricky..

   "Built-in f(x)", see Beck ...

   Options Database Keys:

  Notes:
     Prox formulas are:
  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_Prox(Tao tao)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscNew(&proxP));
  PetscCall(PetscNew(&proxP->L1));

  tao->ops->setup          = TaoSetUp_Prox;
  tao->ops->solve          = TaoSolve_Prox;
  tao->ops->view           = TaoView_Prox;
  tao->ops->setfromoptions = TaoSetFromOptions_Prox;
  tao->ops->destroy        = TaoDestroy_Prox;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  tao->data                = (void *)proxP;
  proxP->y                 = NULL;
  proxP->eta               = 0.1;
  proxP->strategy          = TAO_PROX_STRATEGY_DEFAULT;
  proxP->stepsize          = 1.;
  proxP->stepsize_old      = 0.;
  proxP->orig_objP         = NULL;
  proxP->orig_objgradP     = NULL;
  proxP->orig_gradP        = NULL;
  proxP->orig_hessP        = NULL;
  proxP->ops->orig_obj     = NULL;
  proxP->ops->orig_objgrad = NULL;
  proxP->ops->orig_grad    = NULL;
  proxP->ops->orig_hess    = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxSetStepSize(Tao tao, PetscReal step)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->stepsize = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxGetStepSize(Tao tao, PetscReal *step)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *step = proxP->stepsize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxSetVM(Tao tao, Mat vm)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->vm = vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxGetVM(Tao tao, Mat *vm)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *vm = proxP->vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxSetInitialVector(Tao tao, Vec y) 
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxGetInitialVector(Tao tao, Vec *y) 
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *y = proxP->y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TaoGetProxParentTao - Gets pointer to parent `TAOProx`, used by inner subsolver.

   Collective

   Input Parameter:
. tao - the `Tao` context

   Output Parameter:
. prox_tao - the parent `Tao` context

   Level: advanced

.seealso: `TAOProx`
@*/
PetscErrorCode TaoGetProxParentTao(Tao tao, Tao *prox_tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscObjectQuery((PetscObject)tao, "TaoGetProxParentTao_ADMM", (PetscObject *)prox_tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}



