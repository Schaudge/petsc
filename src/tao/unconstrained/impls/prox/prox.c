#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petscmath.h>
#include <../src/tao/unconstrained/impls/prox/prox.h>

//User means solving it via user specified types, e.g., TAOCG. 
//Metric would internally add MR.
//For User, TAOTYPE may not be TAOPROX.
//TODO do I even need bregman here???
const char *const TaoMetricTypes[] = {"USER", "L1", "L2", "BREGMAN", "TaoMetricType", "TAO_METRIC_", NULL};

const char *const TaoPROXStrategies[] = {"STRATEGY_DEFAULT", "STRATEGY_ADAPTIVE", "STRATEGY_VM", "TaoPROXStrategy", "TAO_PROX_", NULL};
const char *const TaoPROXTypes[] = {"DEFAULT", "L1", "TaoPROXType", "TAO_PROX_", NULL};


static PetscErrorCode AddMoreauRegObj(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  TAO_PROX *proxP  =  (TAO_PROX *)&ptr;
  PetscReal temp;

  PetscFunctionBegin;
  //TODO check MetricType stuff
  /* Adding |x-y|_2^2 */
  /* Ignore VM for now */
  /* Scalar weight */
  //TODO what does it mean if tao is subtao???
  PetscCall((proxP->ops->orig_obj)(tao, X, f, proxP->orig_objP));

  if (tao->metric_type == TAO_METRIC_TYPE_L2) {
    PetscCall(VecWAXPY(proxP->workvec1, -1., proxP->y, X));
    PetscCall(VecNorm(proxP->workvec1,NORM_2, &temp));
    temp = PetscPowReal(temp,2);
    *f += (proxP->stepsize/2)*temp;
  } else if (tao->metric_type == TAO_METRIC_TYPE_USER) {
    PetscCall((tao->ops->computemetricandgradient)(tao, X, proxP->y, &temp, NULL, tao->user_metricP));
    *f += (proxP->stepsize)*temp;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddMoreauRegGrad(Tao tao, Vec X, Vec G, void *ptr)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  //TODO check MetricType stuff
  PetscCall((proxP->ops->orig_grad)(tao, X, G, proxP->orig_gradP));
  PetscCall(VecWAXPY(proxP->workvec1, -1., proxP->y, X));
  PetscCall(VecAXPY(G, proxP->stepsize, proxP->workvec1)); 
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddMoreauRegObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;
  PetscReal temp;

  PetscFunctionBegin;
  //TODO check MetricType stuff
  PetscCall((proxP->ops->orig_objgrad)(tao, X, f, G, proxP->orig_objgradP));
  PetscCall(VecWAXPY(proxP->workvec1, -1., proxP->y, X));
  PetscCall(VecNorm(proxP->workvec1,NORM_2, &temp));
  temp = PetscPowReal(temp,2);

  *f += (proxP->stepsize/2)*temp;
  PetscCall(VecAXPY(G, proxP->stepsize, proxP->workvec1)); 
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddMoreauRegHess(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  //TODO check MetricType stuff
  PetscCall((proxP->ops->orig_hess)(tao, X, H, Hpre, proxP->orig_hessP));
  if (proxP->stepsize != proxP->stepsize_old) {
    PetscCall(MatShift(H, proxP->stepsize - proxP->stepsize_old));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* Updates Moreau Regularization to the given objective and gradient and Hessian */
/* Ignoring VM for now ... */
static PetscErrorCode AddMoreauReg(Tao tao)
{
  TAO_PROX *proxP  = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (proxP->y == NULL) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Need to set y vector for TAOPROX first.");
  }

  if (tao->ops->computeobjective) {
    proxP->orig_objP     = tao->user_objP;
    proxP->ops->orig_obj = tao->ops->computeobjective;

    if (tao->user_objP) proxP->subsolver->user_objP = tao->user_objP;
    if (tao->ops->computeobjective) proxP->subsolver->ops->computeobjective = tao->ops->computeobjective;

    /* Adding MR */
    PetscCall(TaoSetObjective(tao, AddMoreauRegObj, &proxP));
    PetscCall(TaoSetObjective(proxP->subsolver, AddMoreauRegObj, &proxP));
  }

  if (tao->ops->computegradient) {
    proxP->orig_gradP     = tao->user_gradP;
    proxP->ops->orig_grad = tao->ops->computegradient;

    if (tao->user_gradP) proxP->subsolver->user_objP = tao->user_gradP;
    if (tao->ops->computegradient) proxP->subsolver->ops->computegradient= tao->ops->computegradient;

    if (tao->gradient) {
      PetscCall(PetscObjectReference((PetscObject)tao->gradient));
      PetscCall(VecDestroy(&proxP->subsolver->gradient));
      proxP->subsolver->gradient = tao->gradient;
    }
    PetscCall(TaoSetGradient(tao, tao->gradient, AddMoreauRegGrad, &proxP));
    PetscCall(TaoSetGradient(proxP->subsolver, tao->gradient, AddMoreauRegGrad, &proxP));
  }

  if (tao->ops->computeobjectiveandgradient) {
    proxP->orig_objgradP     = tao->user_objgradP;
    proxP->ops->orig_objgrad = tao->ops->computeobjectiveandgradient;

    if (tao->user_objgradP) proxP->subsolver->user_objgradP = tao->user_objgradP;
    if (tao->ops->computeobjectiveandgradient) proxP->subsolver->ops->computeobjectiveandgradient= tao->ops->computeobjectiveandgradient;

    if (tao->gradient) {
      PetscCall(PetscObjectReference((PetscObject)tao->gradient));
      PetscCall(VecDestroy(&proxP->subsolver->gradient));
      proxP->subsolver->gradient = tao->gradient;
    }
    PetscCall(TaoSetObjectiveAndGradient(tao, tao->gradient, AddMoreauRegObjGrad, &proxP));
    PetscCall(TaoSetObjectiveAndGradient(proxP->subsolver, tao->gradient, AddMoreauRegObjGrad, &proxP));
  }

  if (tao->ops->computehessian) {
    proxP->orig_hessP     = tao->user_hessP;
    proxP->ops->orig_hess = tao->ops->computehessian;
    proxP->H_orig         = tao->hessian;
    proxP->H_pre_orig     = tao->hessian_pre;

    if (tao->user_hessP) proxP->subsolver->user_hessP = tao->user_hessP;
    if (tao->ops->computehessian) proxP->subsolver->ops->computehessian= tao->ops->computehessian;

    if (tao->hessian) {
      PetscCall(PetscObjectReference((PetscObject)tao->hessian));
      PetscCall(MatDestroy(&proxP->subsolver->hessian));
      proxP->subsolver->hessian = tao->hessian;
    }
    if (tao->hessian_pre) {
      PetscCall(PetscObjectReference((PetscObject)tao->hessian_pre));
      PetscCall(MatDestroy(&proxP->subsolver->hessian_pre));
      proxP->subsolver->hessian_pre = tao->hessian_pre;
    }
    PetscCall(TaoSetHessian(tao, tao->hessian, tao->hessian_pre, AddMoreauRegHess, &proxP));
    PetscCall(TaoSetHessian(proxP->subsolver, tao->hessian, tao->hessian_pre, AddMoreauRegHess, &proxP));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_PROX(Tao tao)
{
  TAO_PROX  *proxP = (TAO_PROX *)tao->data;
  PetscReal  step  = 1.0, f, gnorm;

  PetscFunctionBegin;

  /*  Check convergence criteria */
  if (proxP->y == NULL) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Need to set y vector for TAOPROX first.");
  }

//  if (proxP->type == TAO_PROX_TYPE_DEFAULT) {
//  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
//  PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
//  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
//
//
//  tao->reason = TAO_CONTINUE_ITERATING;
//  PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
//  PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
//  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
//  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);


  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);

    /*  Save the current gradient information */
    PetscCall(VecCopy(tao->solution, proxP->X_old));
    PetscCall(VecCopy(tao->gradient, proxP->G_old));

    switch (proxP->type) {
    case TAO_PROX_TYPE_DEFAULT:
      {        
        /* Case 1 and 2 */
        PetscCall(TaoSolve(proxP->subsolver));
        tao->reason = proxP->subsolver->reason;
      }
      break;
    case TAO_PROX_TYPE_L1:
      {
        /* Case 3 */      
        //TODO in this case, should I care about subsolver tao??
        PetscCall(TaoSoftThreshold(proxP->y, proxP->L1->lb, proxP->L1->ub, tao->solution));
        tao->reason = TAO_CONVERGED_USER;
      }
      break;
    default:
      break; 
    }

    /* Update stepsize / VM  TODO */

    /*  Check for termination */
    tao->niter++;
//    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
//    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
//    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
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
  if (!proxP->workvec1) PetscCall(VecDuplicate(tao->solution, &proxP->workvec1));
  if (!proxP->workvec1) PetscCall(VecDuplicate(tao->solution, &proxP->workvec1));
  PetscCall(TaoSetSolution(proxP->subsolver, tao->solution));
  PetscCall(AddMoreauReg(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_PROX(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&proxP->X_old));
    PetscCall(VecDestroy(&proxP->G_old));
    PetscCall(VecDestroy(&proxP->workvec1));
    PetscCall(VecDestroy(&proxP->y));
  }
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_PROX(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Proximal algorithm optimization");
  PetscCall(PetscOptionsEnum("-tao_prox_strategy", "TAOPROX update strategies", "TaoPROXStrategy", TaoPROXStrategies, (PetscEnum)proxP->strategy, (PetscEnum *)&proxP->strategy, NULL));
  PetscCall(PetscOptionsEnum("-tao_prox_type", "TAOPROX solver type", "TaoPROXType", TaoPROXTypes, (PetscEnum)proxP->type, (PetscEnum *)&proxP->type, NULL));
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "PROX Strategy: %s\n", TaoPROXStrategies[proxP->strategy]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PROX Type: %s\n", TaoPROXTypes[proxP->type]));
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
     PROX formulas are:
  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_PROX(Tao tao)
{
  TAO_PROX     *proxP;

  PetscFunctionBegin;
  PetscCall(PetscNew(&proxP));
  PetscCall(PetscNew(&proxP->L1));

  tao->ops->setup          = TaoSetUp_PROX;
  tao->ops->solve          = TaoSolve_PROX;
  tao->ops->view           = TaoView_PROX;
  tao->ops->setfromoptions = TaoSetFromOptions_PROX;
  tao->ops->destroy        = TaoDestroy_PROX;

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

  proxP->L1->lb = 0;
  proxP->L1->ub = 0;

  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &proxP->subsolver));
  PetscCall(TaoSetOptionsPrefix(proxP->subsolver, "prox_subsolver_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)proxP->subsolver, (PetscObject)tao, 1));
  PetscCall(TaoSetType(proxP->subsolver, TAONM));
  
  PetscCall(PetscObjectCompose((PetscObject)proxP->subsolver, "TaoGetPROXParentTao_PROX", (PetscObject)tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXGetSubsolver(Tao tao, Tao *subsolver)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *subsolver= proxP->subsolver;
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

PETSC_EXTERN PetscErrorCode TaoPROXSetSoftThreshold(Tao tao, PetscReal lb, PetscReal ub)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->type   = TAO_PROX_TYPE_L1;
  proxP->L1->lb = lb;
  proxP->L1->ub = ub;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXSetInitialVector(Tao tao, Vec y) 
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXGetInitialVector(Tao tao, Vec *y) 
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *y = proxP->y;
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

//TODO
// does this func belong here?
//
//Technically, TAO here isn't really TAOPROX. its whatever, really. 
//Actually, onlything i need is callback for obj/obj,grad/ and its pointers... 
PETSC_EXTERN PetscErrorCode TaoApplyProximalMap(Tao tao, PetscReal lambda, Mat VM, Vec y, Vec x)
{
  TaoType   tao_type;
  PetscBool is_prox;
  PetscFunctionBegin;

  PetscCall(TaoGetType(tao, &tao_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));

  if (!is_prox) {
    if (tao->metric_subtao) {
    // subtao is already created. probably second+ call to this func?
      //TODO reference counter thing ?? add tao destroy for master taodestroy
      TaoType sub_type;
      PetscBool is_sub_prox;
      PetscCall(TaoGetType(tao->metric_subtao, &sub_type));
      PetscCall(PetscObjectTypeCompare((PetscObject)tao->metric_subtao, TAOPROX, &is_sub_prox));
      if (!is_sub_prox) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "METRIC SUBTAO TYPE IS NOT PROX.");
    } else {
      PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &tao->metric_subtao));
      PetscCall(TaoSetType(tao->metric_subtao, TAOPROX));
    }
      PetscCall(TaoSetSolution(tao->metric_subtao, x));
      PetscCall(TaoPROXSetInitialVector(tao->metric_subtao, y));
      PetscCall(TaoSetFromOptions(tao->metric_subtao));
  }
    PetscCall(TaoSolve(tao->metric_subtao));
  PetscFunctionReturn(PETSC_SUCCESS);
}
