#include <petsc/private/taopdimpl.h>
#include <../src/tao/pd/impls/l2/l2.h>

static PetscErrorCode TaoPDDestroy_L2(TaoPD pd)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(pd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDSetFromOptions_L2(TaoPD pd, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "L2 PD options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDComputeObjective_L2(TaoPD pd, Vec X, PetscReal *f, void *ctx)
{
  TaoPD_L2 *l2ctx = (TaoPD_L2 *)pd->data;
  Vec       y;
  Mat       vm;

  PetscFunctionBegin;
  PetscCall(TaoPDGetCentralVector(pd, &y));

  vm = l2ctx->vm;
  if (y) {
    PetscCall(VecWAXPY(pd->workvec, -1., y, X));
    if (l2ctx->vm) {
      PetscCall(MatMult(vm, pd->workvec, pd->workvec2));
      PetscCall(VecTDot(pd->workvec, pd->workvec2, f));
    } else {
      PetscCall(VecNorm(pd->workvec, NORM_2, f));
    }
  } else {
    if (l2ctx->vm) {
      PetscCall(MatMult(vm, X, pd->workvec));
      PetscCall(VecTDot(X, pd->workvec, f));
    } else {
      PetscCall(VecNorm(X, NORM_2, f));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDComputeGradient_L2(TaoPD pd, Vec X, Vec G, void *ctx)
{
  Mat       vm;
  Vec       y;
  TaoPD_L2 *l2ctx = (TaoPD_L2 *)pd->data;

  PetscFunctionBegin;
  PetscCall(TaoPDGetCentralVector(pd, &y));
  vm = l2ctx->vm;
  if (y) {
    if (vm) {
      PetscCall(VecAXPBYPCZ(pd->workvec, 2, -2, 0, X, y));
      PetscCall(MatMult(vm, pd->workvec, G));
    } else {
      PetscCall(VecAXPBYPCZ(G, 2, -2, 0, X, y));
    }
  } else {
    if (vm) {
      PetscCall(VecAXPBY(pd->workvec, 2, 0., X));
      PetscCall(MatMult(vm, pd->workvec, G));
    } else {
      PetscCall(VecAXPBY(G, 2, 0., X));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDComputeObjectiveAndGradient_L2(TaoPD pd, Vec X, PetscReal *f, Vec G, void *ctx)
{
  Mat       vm;
  Vec       y;
  TaoPD_L2 *l2ctx = (TaoPD_L2 *)pd->data;

  PetscFunctionBegin;
  PetscCall(TaoPDGetCentralVector(pd, &y));
  vm = l2ctx->vm;

  if (y) {
    if (vm) {
      PetscCall(VecAXPBYPCZ(pd->workvec, 2, -2, 0, X, y));
      PetscCall(MatMult(vm, pd->workvec, G));
      PetscCall(VecTDot(pd->workvec, pd->workvec2, f));
    } else {
      PetscCall(VecAXPBYPCZ(G, 1, -1, 0, X, y));
      PetscCall(VecTDot(G, G, f));
      PetscCall(VecScale(G, 2.));
    }
  } else {
    if (vm) {
      PetscCall(MatMult(vm, X, G));
      PetscCall(VecTDot(X, G, f));
      PetscCall(VecScale(G, 2));
    } else {
      PetscCall(VecTDot(X, X, f));
      PetscCall(VecAXPBY(G, 2, 0., X));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDApplyProximalMap_L2(TaoPD pd0, TaoPD pd1, PetscReal lambda, Vec y, Vec x, void *ctx)
{
  TaoPDType pd0_type;
  PetscBool is_0_l2;

  PetscFunctionBegin;
  PetscCall(TaoPDGetType(pd0, &pd0_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)pd0, TAOPDL2, &is_0_l2));
  PetscAssert(is_0_l2, PetscObjectComm((PetscObject)pd0), PETSC_ERR_USER, "L2 Square does not have proximal map.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoPDL2SetContext(TaoPD pd, Mat vm)
{
  TaoPD_L2 *ctx = (TaoPD_L2 *)pd->data;

  PetscFunctionBegin;
  ctx->vm = vm;
  PetscCall(PetscObjectReference((PetscObject)vm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO Need to implement Hessian Routine */

PETSC_EXTERN PetscErrorCode TaoPDCreate_L2_Private(TaoPD pd)
{
  TaoPD_L2 *ctx;

  PetscFunctionBegin;
  PetscAssertPointer(pd, 1);
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  ctx->vm  = NULL;
  pd->data = (void *)ctx;

  pd->ops->destroy                     = TaoPDDestroy_L2;
  pd->ops->setfromoptions              = TaoPDSetFromOptions_L2;
  pd->ops->computeobjective            = TaoPDComputeObjective_L2;
  pd->ops->computegradient             = TaoPDComputeGradient_L2;
  pd->ops->computeobjectiveandgradient = TaoPDComputeObjectiveAndGradient_L2;
  pd->ops->applyproximalmap            = TaoPDApplyProximalMap_L2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Doing Create, SetType, SetContext */
PETSC_EXTERN PetscErrorCode TaoPDCreate_L2(MPI_Comm comm, TaoPD *pd, Vec y, Mat vm)
{
  PetscFunctionBegin;
  PetscCall(TaoPDCreate(comm, pd));
  PetscCall(TaoPDSetType(*pd, TAOPDL2));
  PetscCall(TaoPDSetCentralVector(*pd, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}
