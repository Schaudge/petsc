#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/snesimpl.h>

const char *const TaoMetricTypes[] = {"TYPE_USER", "TYPE_L1", "TYPE_L2", "TYPE_DIAGONAL", "TYPE_AFFINE", "TaoMetricType", "TAO_METRIC_", NULL};



/* Setting callback function for Metric.
 *
 * PetscErrorCode func(Tao tao, Vec x, Vec y, PetscReal *f, Vec g, void *ctx)
 * Vec x,y: input. y should be set before hand.
 *
 * f: objective value of the metric 
 * g: gradient of the metric.
 *
 * TODO should we support Hessian?
 * TODO should we support initial compute of thigns?
 * D(x,y)
 *
 * e.g., Bregman: D_h(x,y) = h(x) - h(y) - \langle \nabla h(y), x-y \rangle.
 *
 * If h(.) = \|. \|_2^2, above becomes \|x-y\|_2^2  */
PetscErrorCode TaoSetMetricRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, Vec, PetscReal *, Vec , void *), void *ctx)
{
  PetscFunctionBegin;
  if (ctx) tao->user_metricP = ctx;
  if (func) tao->ops->computemetricandgradient = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoGetMetricRoutine(Tao tao, PetscErrorCode (**func)(Tao, Vec, Vec, PetscReal *, Vec , void *), void **ctx)
{
  PetscFunctionBegin;
  if (ctx) *ctx = tao->user_metricP;
  if (func) *func = tao->ops->computemetricandgradient;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoSetMetricType(Tao tao, TaoMetricType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(tao, type, 2);
  //TODO perhaps more boilerplates/checks here???
  tao->metric_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoGetMetricType(Tao tao, TaoMetricType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  //TODO what if its not set or something like that?
  *type = tao->metric_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode TaoAddMoreauRegObjGrad_Private(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Vec y, workvec;
  PetscReal stepsize, temp;
  PetscFunctionBegin;

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  PetscCall((tao->MR_internal->ops->computeobjectiveandgradient)(tao, X, f, G, tao->MR_internal->orig_objgradP));
  switch (tao->metric_type) {
  case TAO_METRIC_TYPE_L2:
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecNorm(workvec,NORM_2, &temp));
    temp = PetscPowReal(temp,2);
    PetscCall(VecAXPY(G, stepsize, workvec)); 
    break;
  case TAO_METRIC_TYPE_L1:
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecNorm(workvec,NORM_1, &temp));
    //TODO gradient???
    break;
  case TAO_METRIC_TYPE_USER:
    PetscCall((tao->ops->computemetricandgradient)(tao, X, y, &temp, workvec, tao->user_metricP));
    PetscCall(VecAXPY(G, stepsize, workvec)); 
    break;
  default:
    temp = 0; //TODO
    break;
  }
  *f += stepsize/2*temp;
  PetscFunctionReturn(PETSC_SUCCESS);

}

static PetscErrorCode TaoAddMoreauRegGrad_Private(Tao tao, Vec X, Vec G, void *ptr)
{
  Vec y, workvec;
  PetscReal stepsize;
  PetscFunctionBegin;

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  PetscCall((tao->MR_internal->ops->computegradient)(tao, X, G, tao->MR_internal->orig_gradP));
  switch (tao->metric_type) {
  case TAO_METRIC_TYPE_L2:
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecAXPY(G, stepsize, workvec)); 
    break;
  case TAO_METRIC_TYPE_L1:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "L1 gradient no-op");
    break;
  case TAO_METRIC_TYPE_USER://TODO so something about metric to subtao stuff
    PetscCall((tao->ops->computemetricandgradient)(tao, X, y, NULL, G, tao->user_metricP));
    break;
  default:
    break;          
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoAddMoreauRegObj_Private(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  Vec y, workvec;
  PetscReal temp, stepsize;
  PetscFunctionBegin;

  PetscCall((tao->MR_internal->ops->computeobjective)(tao, X, f, tao->MR_internal->orig_objP));

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  switch (tao->metric_type) {
  case TAO_METRIC_TYPE_L2:
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecNorm(workvec,NORM_2, &temp));
    temp = PetscPowReal(temp,2);
    break;
  case TAO_METRIC_TYPE_L1:
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecNorm(workvec,NORM_1, &temp));
    break;
  case TAO_METRIC_TYPE_USER:
    PetscCall((tao->ops->computemetricandgradient)(tao, X, y, &temp, NULL, tao->user_metricP));
    break;
  default:
    temp = 0;//TODO
    break;          
  }
  *f += (stepsize/2)*temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* If given Tao is, say, f(x), 
 * this function will create subtao  f(x) + g(x,y)
 *
 * where g(x,y) depends on metric type. 
 * if L1, g = |x-y|_1
 * if L2, g = |x-y|_2^2
 * if user, g is given by user.  */
//TODO issue with workvec??
PetscErrorCode TaoAddMoreauRegularizer(Tao tao, Tao subtao, Vec y)
{
  PetscFunctionBegin;

  if (y == NULL) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Need to set y vector for Proximal map.");
  }

  subtao->MR_internal->y = y;
  PetscCall(VecDuplicate(y, &subtao->MR_internal->workvec));

  if (tao->ops->computeobjective) {
    subtao->MR_internal->orig_objP             = tao->user_objP;
    subtao->MR_internal->ops->computeobjective = tao->ops->computeobjective;

    /* Adding MR */
    PetscCall(TaoSetObjective(subtao, TaoAddMoreauRegObj_Private, &tao->user_objP));
  }

  if (tao->ops->computegradient) {
    subtao->MR_internal->orig_gradP           = tao->user_gradP;
    subtao->MR_internal->ops->computegradient = tao->ops->computegradient;

    if (tao->gradient) {
      PetscCall(PetscObjectReference((PetscObject)tao->gradient));
      PetscCall(VecDestroy(&subtao->gradient));
      subtao->gradient = tao->gradient;
    }
    PetscCall(TaoSetGradient(subtao, tao->gradient, TaoAddMoreauRegGrad_Private, &tao->user_gradP));
  }

  if (tao->ops->computeobjectiveandgradient) {
    subtao->MR_internal->orig_objgradP                    = tao->user_objgradP;
    subtao->MR_internal->ops->computeobjectiveandgradient = tao->ops->computeobjectiveandgradient;

    if (tao->gradient) {
      PetscCall(PetscObjectReference((PetscObject)tao->gradient));
      PetscCall(VecDestroy(&subtao->gradient));
      subtao->gradient = tao->gradient;
    }
    PetscCall(TaoSetObjectiveAndGradient(subtao, tao->gradient, TaoAddMoreauRegObjGrad_Private, &tao->user_gradP));
  }

  //TODO should we bother with hessian?
#if 0  
  if (tao->ops->computehessian) {
    proxP->orig_hessP     = tao->user_hessP;
    proxP->ops->orig_hess = tao->ops->computehessian;
    proxP->H_orig         = tao->hessian;
    proxP->H_pre_orig     = tao->hessian_pre;

    if (tao->user_hessP) subtao->user_hessP = tao->user_hessP;
    if (tao->ops->computehessian) subtao->ops->computehessian= tao->ops->computehessian;

    if (tao->hessian) {
      PetscCall(PetscObjectReference((PetscObject)tao->hessian));
      PetscCall(MatDestroy(&subtao->hessian));
      subtao->hessian = tao->hessian;
    }
    if (tao->hessian_pre) {
      PetscCall(PetscObjectReference((PetscObject)tao->hessian_pre));
      PetscCall(MatDestroy(&subtao->hessian_pre));
      subtao->hessian_pre = tao->hessian_pre;
    }
    PetscCall(TaoSetHessian(tao, tao->hessian, tao->hessian_pre, AddMoreauRegHess, &proxP));
    PetscCall(TaoSetHessian(subtao, tao->hessian, tao->hessian_pre, AddMoreauRegHess, &proxP));
  }
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO
// does this func belong here?
//
//Technically, TAO here isn't really TAOPROX. its whatever, really. 
//Actually, onlything i need is callback for obj/obj,grad/ and its pointers... 
PETSC_EXTERN PetscErrorCode TaoApplyProximalMap(Tao tao, PetscReal lambda, Vec y, Vec x)
{
  TaoType   tao_type;
  PetscBool is_prox;
  PetscFunctionBegin;

  /* if lambda == 0.0, just call TaoSolve.
   * if y == NULL, y == 0 */
  PetscCall(TaoGetType(tao, &tao_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));

  if (is_prox) {
    /* Proximal Case */
    PetscUseTypeMethod(tao, applyproximalmap, lambda, y, x);
  } else {
    /* Type is non-prox. */
    // first time this called. need to initialize subtao
    if (tao->proximalmap_subtao == NULL) {
      TaoType type;
      TaoGetType(tao, &type);
      PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &tao->proximalmap_subtao));
      PetscCall(TaoSetType(tao->proximalmap_subtao, type));
      PetscCall(TaoSetSolution(tao, x));
      PetscCall(TaoAddMoreauRegularizer(tao, tao->proximalmap_subtao, y));
    }
    PetscCall(TaoSolve(tao));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
