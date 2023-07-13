#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/snesimpl.h>


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

#if 0
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
#endif

static PetscErrorCode TaoAddMoreauRegHess_Private(Tao tao, Vec X, Mat H, Mat H_pre, void *ptr)
{
  Vec y, workvec;
  PetscReal stepsize, temp;
  PetscFunctionBegin;

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  PetscCall((tao->MR_internal->ops->computehessian)(tao, X, H, H_pre, tao->MR_internal->orig_hessP));
  switch (tao->metric_type) {
  case TAO_METRIC_TYPE_L2:
    PetscCall(MatShift(H,stepsize));//TODO make sure H is "flushed every time its called to avoid aI stacking?
    break;
  case TAO_METRIC_TYPE_L1:
    break;
  case TAO_METRIC_TYPE_USER:
    //TODO
    break;
  default:
    temp = 0; //TODO
    break;
  }
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
  PetscBool isl2;

  //PetscErrorCode (*metric_xxx)(Tao, PetscReal *, Vec, Vec, PetscReal *, Vec, Mat, Mat, void *);

  PetscFunctionBegin;

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  PetscCall((tao->MR_internal->ops->computegradient)(tao, X, G, tao->MR_internal->orig_gradP));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &isl2));
  if (isl2) {
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecAXPY(G, stepsize, workvec)); 
  } else {
   
  }
    //SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "L1 gradient no-op");
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

  if (tao->ops->computehessian) {
    subtao->MR_internal->orig_hessP          = tao->user_hessP;
    subtao->MR_internal->ops->computehessian = tao->ops->computehessian;
    subtao->MR_internal->H                   = tao->hessian;
    subtao->MR_internal->H_pre               = tao->hessian_pre;

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
    PetscCall(TaoSetHessian(subtao, tao->hessian, tao->hessian_pre, TaoAddMoreauRegHess_Private, &tao->user_hessP));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
