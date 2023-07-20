#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

PetscBool         TaoProxRegisterAllCalled   = PETSC_FALSE;
PetscBool         TaoMetricRegisterAllCalled = PETSC_FALSE;
PetscFunctionList TaoProxList                = NULL;
PetscFunctionList TaoMetricList              = NULL;

//TODO should most of the code here be in proxutil.c? or here?

/*@C
   TaoProxSetType - Sets the `TaoProxType` for the prox routine.

   Collective

   Input Parameters:
+  tao - the `Tao` solver context
-  type - a known method

   Options Database Key:
.  -tao_prox_type <type> - Sets the method; use -help for a list
   of available methods (for instance, "-tao_prox_type prox_l2")

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoProxGetType()`, `TaoProxType`
@*/
PetscErrorCode TaoProxSetType(Tao tao, TaoProxType type)
{
  PetscErrorCode (*prox_xxx)(Tao, PetscReal, Vec, Vec, void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  //TODO issame check needed or not?

  PetscCall(PetscFunctionListFind(TaoProxList, type, (void (**)(void)) & prox_xxx));
  PetscCheck(prox_xxx, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested Tao prox type %s", type);
  tao->prox_type             = type;
  tao->ops->applyproximalmap = prox_xxx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoProxGetType - Gets the current `TaoProxType` being used in the `Tao` object

   Not Collective

   Input Parameter:
.  tao - the `Tao` solver context

   Output Parameter:
.  type - the `TaoProxType`

   Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoProxType`, `TaoProxGetType()`
@*/

PetscErrorCode TaoProxGetType(Tao tao, TaoProxType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = tao->prox_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoProxRegister - Adds a method to the Tao package for minimization.

   Not Collective

   Input Parameters:
+  sname - name of a new user-defined proximal mapping
.  func -  a callback for proximal mapping solver
-  ctx - [optional] user-defined context for private data for proximal solver 
         (may be 'NULL')

   
  Calling sequence of `func`:
$ PetscErrorCode func(Tao tao, PetscReal s, Vec y, Vec x, void *ctx);
+ tao - the optimization object
. s - step size 
. y - centraling vector term
. x - solution vector (output)
- ctx - [optional] user-defined function context

   Sample usage:
.vb
   TaoProxRegister("my_prox_solver", MyProxCallBack);
.ve

   Then, your proximal solver can be chosen with the procedural interface via
$     TaoProxSetType(tao, "my_prox_solver")
   or at runtime via the option
$     -tao_prox_type my_prox_solver

   Level: advanced

   Note:
   `TaoProxRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](ch_tao), `Tao`, `TaoSetType()`, `TaoProxRegisterAll()`, `TaoProxRegisterDestroy()`
@*/
PetscErrorCode TaoProxRegister(const char name[], PetscErrorCode (*func)(Tao, PetscReal, Vec, Vec, void *))
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoProxList, name, (void (*)(void))func));
  //TODO add context somehow, or is it even necessary?
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoProxRegisterDestroy - Frees the list of proximal solvers that were
   registered by `TaoProxRegister()`.

   Not Collective

   Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoProxRegisterAll()`, `TaoProxRegister()`
@*/
PetscErrorCode TaoProxRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoProxList));
  TaoProxRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoAddMoreauRegHess_Private(Tao tao, Vec X, Mat H, Mat H_pre, void *ptr)
{
  PetscReal stepsize;
  PetscBool is_l2, is_diag;
  PetscFunctionBegin;

  stepsize = tao->MR_internal->stepsize;
  PetscCall((tao->MR_internal->ops->computehessian)(tao, X, H, H_pre, tao->MR_internal->orig_hessP));

  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &is_l2));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_DIAG, &is_diag));

  if (is_l2) {
    PetscCall(MatShift(H,stepsize));//TODO make sure H is "flushed every time its called to avoid aI stacking?
  } else if (is_diag) {
    PetscCall(MatDiagonalScale(H, tao->MR_internal->diag_metric, NULL));//TODO make sure H is "flushed every time its called to avoid aI stacking?
  } else {
    Vec y;
    Mat H_work;

    y      = tao->MR_internal->y;
    H_work = tao->MR_internal->H_work;
    //Via outside TaoMetricRegister TODO make sure its not null. petscassert, check??
    //Need to document that NULL needs to be no-op for user....
    PetscCallBack("User provided Metric callback, Hessian part", (*tao->ops->computemetric)(tao, &stepsize, y, X, NULL, NULL, H_work, ptr));
    PetscCall(MatAXPY(H, 1., H_work, UNKNOWN_NONZERO_PATTERN)); 
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoAddMoreauRegObjGrad_Private(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Vec y, workvec;
  PetscReal stepsize, temp;
  PetscBool is_l2, is_diag;
  PetscFunctionBegin;

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  PetscCall((tao->MR_internal->ops->computeobjectiveandgradient)(tao, X, f, G, tao->MR_internal->orig_objgradP));

  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &is_l2));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_DIAG, &is_diag));

  if (is_l2) {
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecNorm(workvec,NORM_2, &temp));
    temp = PetscPowReal(temp,2);
    PetscCall(VecAXPY(G, stepsize, workvec)); 
  } else if (is_diag) {
    Vec diag;
    diag = tao->MR_internal->diag_metric;

    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecPointwiseMult(G, diag, workvec));
    PetscCall(VecDot(G, workvec, &temp));
  } else {
    PetscCallBack("User provided Metric callback, ObjGrad part", (*tao->ops->computemetric)(tao, &stepsize, y, X, &temp, workvec, NULL, ptr));
    PetscCall(VecAXPY(G, stepsize, workvec)); 
  }
  *f += stepsize/2*temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoAddMoreauRegGrad_Private(Tao tao, Vec X, Vec G, void *ptr)
{
  Vec y, workvec;
  PetscReal stepsize;
  PetscBool is_l2, is_diag;

  //PetscErrorCode (*metric_xxx)(Tao, PetscReal *, Vec, Vec, PetscReal *, Vec, Mat, Mat, void *);

  PetscFunctionBegin;

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  PetscCall((tao->MR_internal->ops->computegradient)(tao, X, G, tao->MR_internal->orig_gradP));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &is_l2));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_DIAG, &is_diag));
  if (is_l2) {
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecAXPY(G, stepsize, workvec)); 
  } else if (is_diag) {
    Vec diag;
    diag = tao->MR_internal->diag_metric;

    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecPointwiseMult(G, diag, workvec));
  } else {
    PetscCallBack("User provided Metric callback, ObjGrad part", (*tao->ops->computemetric)(tao, &stepsize, y, X, NULL, workvec, NULL, ptr));
    PetscCall(VecAXPY(G, stepsize, workvec)); 
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoAddMoreauRegObj_Private(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  Vec       y, workvec;
  PetscReal temp, stepsize;
  PetscBool is_l2, is_diag;

  PetscFunctionBegin;
  PetscCall((tao->MR_internal->ops->computeobjective)(tao, X, f, tao->MR_internal->orig_objP));

  y = tao->MR_internal->y;
  workvec = tao->MR_internal->workvec;
  stepsize = tao->MR_internal->stepsize;

  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &is_l2));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_DIAG, &is_diag));

  if (is_l2) {
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecNorm(workvec,NORM_2, &temp));
    temp = PetscPowReal(temp,2);
  } else if (is_diag) {
    Vec diag;
    diag = tao->MR_internal->diag_metric;

    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecPointwiseMult(workvec, diag, workvec));
    PetscCall(VecDot(workvec, workvec, &temp));
  } else {
    PetscCallBack("User provided Metric callback, ObjGrad part", (*tao->ops->computemetric)(tao, &stepsize, y, X, &temp, workvec, NULL, ptr));
  }
  *f += stepsize/2*temp;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* If given Tao is, say, f(x), 
 * this function will create subtao  f(x) + g(x,y)
 *
 * where g(x,y) depends on metric type. 
 * if L1, g = |x-y|_1
 * if L2, g = |x-y|_2^2
 * if user, g is given by user.  */
PetscErrorCode TaoAddMoreauRegularizer(Tao tao, Tao subtao, Vec y)
{
  PetscFunctionBegin;

  if (y == NULL) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Need to set y vector for Proximal map.");
  }

  subtao->MR_internal->y = y;

  if (tao->diag_metric) {
    subtao->MR_internal->diag_metric = tao->diag_metric;
  }
  //TODO issue with workvec??
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
      PetscCall(MatDuplicate(tao->hessian, MAT_DO_NOT_COPY_VALUES, &tao->MR_internal->H_work));
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

PETSC_EXTERN PetscErrorCode TaoApplyProximalMap(Tao tao, PetscReal lambda, Vec y, Vec x)
{
  TaoType   tao_type;
  PetscBool is_prox;
  PetscFunctionBegin;

  /* if lambda == 0.0, just call TaoSolve.
   * if y == NULL, y == 0 */
  PetscCall(TaoGetType(tao, &tao_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
//TODO lock push, etc boilerplates...
  PetscCall(PetscLogEventBegin(TAO_ProximalEval, tao, 0, 0, 0));
  if (is_prox) {
    /* Proximal Case */
    PetscUseTypeMethod(tao, applyproximalmap, lambda, y, x, tao->user_proxP);
  } else {
    /* Type is non-prox. */
    //TODO is below safe? or petsc-legal? PetscTryTypeMethod doesn't give you flag...
    if (tao->ops->applyproximalmap) {
      PetscTryTypeMethod(tao, applyproximalmap, lambda, y, x, tao->user_proxP);
    } else {
      //TODO TaoCopy is fall-back . shallow vs deep
      // first time this called. need to initialize subtao
      if (tao->metric_tao == NULL) {
        Tao           metric_tao;
        TaoType       type;
        TaoMetricType metric_type;
        //ideally it should be TaoCopy
        PetscCall(TaoGetType(tao, &type));
        PetscCall(TaoGetMetricTao(tao, &metric_tao));
        PetscCall(TaoMetricGetType(metric_tao, &metric_type));
        PetscCall(TaoSetSolution(tao, x));
        PetscCall(TaoAddMoreauRegularizer(tao, tao->proximalmap_subtao, y));
      }
      PetscCall(TaoSolve(tao));
    }
  }
  PetscCall(PetscLogEventEnd(TAO_ProximalEval, tao, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoComputeObjectiveAndGradient_MR(Tao tao, Vec X, PetscReal *f, Vec G,  PetscReal *stepsize, Vec y)
{
  Tao metric_tao;
  Vec workvec;
  PetscReal temp;
  TaoMetricType metric_type;

  PetscFunctionBegin;
  //TODO how to deal with MR_internal now?
  //Also with above mr FUNCTION?
  //STEPSIZE?
  workvec = tao->MR_internal->workvec;

  PetscCall(TaoGetMetricTao(tao, &metric_tao));
  PetscCall(TaoMetricGetType(metric_tao, &metric_type));

  PetscCall(TaoComputeObjectiveAndGradient(tao, X, f, G));

  PetscCall(PetscLogEventBegin(TAO_MetricEval, tao, 0, 0, 0));
  if (metric_type == TAO_METRIC_L2) {
    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecNorm(workvec,NORM_2, &temp));
    temp = PetscPowReal(temp,2);
    PetscCall(VecAXPY(G, *stepsize, workvec)); 
  } else if (metric_type == TAO_METRIC_DIAG) {
    Vec diag;
    diag = tao->MR_internal->diag_metric;

    PetscCall(VecWAXPY(workvec, -1., y, X));
    PetscCall(VecPointwiseMult(G, diag, workvec));
    PetscCall(VecDot(G, workvec, &temp));
  } else if (metric_type == TAO_METRIC_KL){
    PetscCall(VecPointwiseDivide(G, X, y));
    PetscCall(VecLog(G));
    PetscCall(VecShift(G,1.));
  } else { 
    //User case. TODO some petscassert on computeobjgrad? do obj grad separately if thats the case?
    PetscCall(TaoComputeObjectiveAndGradient(metric_tao, X, &temp, workvec));
    PetscCall(VecAXPY(G, *stepsize, workvec)); 
  }
  *f += (*stepsize/2)*temp;
  PetscCall(PetscLogEventEnd(TAO_MetricEval, tao, 0, 0, 0));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoMetricRegister - Adds a method to the Tao package for minimization.

   Not Collective

   Input Parameters:
+  sname - name of a new user-defined metric 
.  func -  a callback for metric function, that evalues objective, gradient, and Hessian. 
-  ctx - [optional] user-defined context for private data for metric routine. 
         (may be 'NULL')

   Sample usage:
.vb
   TaoMetricRegister("my_metric_solver", MyMetricCreate);
.ve

   Then, your metricimal solver can be chosen with the procedural interface via
$     TaoMetricSetType(tao, "my_metric_solver")
   or at runtime via the option
$     -tao_metric_type my_metric_solver

   Level: advanced

   Note:
   `TaoMetricRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](ch_tao), `Tao`, `TaoSetType()`, `TaoMetricRegisterAll()`, `TaoMetricRegisterDestroy()`
@*/
PetscErrorCode TaoMetricRegister(const char name[], PetscErrorCode (*func)(Tao))
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoMetricList, name, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode TaoMetricSetContext(Tao tao, void *ptr)
{
  PetscFunctionBegin;
   tao->user_metricP = ptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoMetricGetContext(Tao tao, void *ptr)
{
  PetscFunctionBegin;
  ptr = tao->user_metricP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoMetricRegisterDestroy - Frees the list of metricimal solvers that were
   registered by `TaoMetricRegister()`.

   Not Collective

   Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoMetricRegisterAll()`, `TaoMetricRegister()`
@*/
PetscErrorCode TaoMetricRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoMetricList));
  TaoMetricRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
/*@C
   TaoMetricSetType - Sets the `TaoMetricType` for the metric routine.

   Collective

   Input Parameters:
+  tao - the `Tao` solver context
-  type - a known method

   Options Database Key:
.  -tao_metric_type <type> - Sets the method; use -help for a list
   of available methods (for instance, "-tao_metric_type metric_l2")

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoMetricGetType()`, `TaoMetricType`
@*/
PetscErrorCode TaoMetricSetType(Tao tao, TaoMetricType type)
{
  PetscErrorCode (*metric_xxx)(Tao);
  PetscBool      issame;
  void          *ptr = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscStrcmp(tao->metric_type, type, &issame));//TODO if metric_Type is null, is this still okay?
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoMetricList, type, (void (**)(void)) & metric_xxx));
  PetscCheck(metric_xxx, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested Tao metric type %s", type);
  tao->metric_type        = type;
  tao->user_metricP       = ptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoGetMetricType - Gets the current `TaoMetricType` being used in the `Tao` object

   Not Collective

   Input Parameter:
.  tao - the `Tao` solver context

   Output Parameter:
.  type - the `TaoMetricType`

   Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoMetricType`, `TaoMetricSetType()`
@*/

PetscErrorCode TaoMetricGetType(Tao tao, TaoMetricType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = tao->metric_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
