#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

PetscBool         TaoProxRegisterAllCalled   = PETSC_FALSE;
PetscFunctionList TaoProxList                = NULL;

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

PETSC_EXTERN PetscErrorCode TaoApplyProximalMap(Tao tao, PetscReal lambda, Vec y, Vec x)
{
  PetscFunctionBegin;
  //TODO lock push, etc boilerplates...
  PetscCall(PetscLogEventBegin(TAO_ProximalEval, tao, 0, 0, 0));
  PetscUseTypeMethod(tao, applyproximalmap, lambda, y, x, tao->user_proxP);
  PetscUseTypeMethod(tao, applyproximalmap, lambda, y, x, tao->user_proxP);
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
    PetscCall(VecPointwiseMult(workvec, G, X));
    PetscCall(VecSum(workvec, &temp));
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
