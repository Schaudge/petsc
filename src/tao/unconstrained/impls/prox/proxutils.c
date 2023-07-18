#include <../src/tao/unconstrained/impls/prox/prox.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>

//static PetscErrorCode TaoApplyProximalMap_L1(Tao, PetscReal, Vec, Vec, void*);
//static PetscErrorCode TaoApplyProximalMap_Affine(Tao, PetscReal, Vec, Vec, void*);
//static PetscErrorCode TaoApplyProximalMap_Simplex(Tao, PetscReal, Vec, Vec, void*);



/*@
   TaoPROXGetSubsolver - Retrieve the subsolver being used by `TAOPROX`.

   Input Parameter:
.  tao - the `Tao` context for the `TAOPROX` solver

   Output Parameter:
.  subsolver - the `Tao` context for the subsolver

   Level: advanced

.seealso: `Tao`, `TAOPROX`, `TaoPROXSetSubsolver()`
@*/
PetscErrorCode TaoPROXGetSubsolver(Tao tao, Tao *subsolver)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(subsolver, 2);
  PetscUseMethod(tao, "TaoPROXGetSubsolver_C", (Tao, Tao *), (tao, subsolver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoPROXGetSubsolver_Private(Tao tao, Tao *subsolver)
{
  TAO_PROX *proxP= (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *subsolver = proxP->subsolver;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXSetSoftThreshold(Tao tao, PetscReal lb, PetscReal ub)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->type   = TAOPROX_L1;
  proxP->L1->lb = lb;
  proxP->L1->ub = ub;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPROXSetAffine(Tao tao, Vec a)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->type      = TAOPROX_L1;
  proxP->affine->a = a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoApplyProximalMap_Affine(Tao tao, PetscReal lambda, Vec y, Vec x, void* ptr)
{
  TaoMetricType metric_type;
  PetscBool     is_prox, is_l2, is_diag;

  PetscFunctionBegin;
  PetscCall(TaoGetMetricType(tao, &metric_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &is_l2));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_DIAG, &is_diag));

  if (!is_prox) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_Affine is called, but TAOTYPE is not TAOPROX..");
  }

  PetscCall(PetscLogEventBegin(TAO_MetricEval, tao, 0, 0, 0));
  if (is_l2 || is_diag) {
      TAO_PROX *proxP= (TAO_PROX *)tao->data;
      Vec a;
      a = proxP->affine->a;
      PetscCall(VecWAXPY(x, -1., a, y));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_Affine cannot support user type metric.");
  }
  PetscCall(PetscLogEventEnd(TAO_MetricEval, tao, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoApplyProximalMap_L1(Tao tao, PetscReal lambda, Vec y, Vec x, void* ptr)
{
  TaoMetricType metric_type;
  PetscBool     is_prox, is_l2, is_diag;

  PetscFunctionBegin;
  PetscCall(TaoGetMetricType(tao, &metric_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &is_l2));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_DIAG, &is_diag));

  if (!is_prox) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_L1 is called, but TAOTYPE is not TAOPROX..");
  }

  PetscCall(PetscLogEventBegin(TAO_MetricEval, tao, 0, 0, 0));
  if (is_l2) {
    TAO_PROX *proxP= (TAO_PROX *)tao->data;
    if (&(proxP->L1->lb) == NULL || &(proxP->L1->ub) == NULL) {
      PetscCall(TaoSoftThreshold(y, -lambda, lambda, x));
    } else {
      PetscCall(TaoSoftThreshold(y, proxP->L1->lb, proxP->L1->ub, x));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_Affine cannot support user type metric.");
  }
  PetscCall(PetscLogEventEnd(TAO_MetricEval, tao, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoApplyProximalMap_Simplex(Tao tao, PetscReal lambda, Vec y, Vec x, void* ptr)
{
  TaoMetricType metric_type;
  PetscBool     is_prox, is_l2, is_diag;

  PetscFunctionBegin;
  PetscCall(TaoGetMetricType(tao, &metric_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_L2, &is_l2));
  PetscCall(PetscStrcmp(tao->metric_type, TAOMETRIC_DIAG, &is_diag));

  if (!is_prox) {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_Simplex is called, but TAOTYPE is not TAOPROX..");
  }

  //TODO implement simplex proj
  PetscFunctionReturn(PETSC_SUCCESS);
}

