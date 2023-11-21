#include <petsc/private/taopdimpl.h>
#include <../src/tao/pd/impls/l1/l1.h>

static PetscErrorCode TaoPDContextDestroy_L1(TaoPD pd)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(pd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDSetFromOptions_L1(TaoPD pd, PetscOptionItems *PetscOptionsObject)
{
  TaoPD_L1 *ctx = (TaoPD_L1 *)pd->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoPD Simplex options");
  PetscCall(PetscOptionsReal("-tao_pd_l1_lb", "PD SoftThreshold lower bound.", "", ctx->lb, &ctx->lb, NULL));
  PetscCall(PetscOptionsReal("-tao_pd_l1_ub", "PD SoftThreshold upper bound.", "", ctx->ub, &ctx->ub, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoPDL1SetContext(TaoPD pd, PetscReal lb, PetscReal ub)
{
  TaoPD_L1 *ctx = (TaoPD_L1 *)pd->data;

  PetscFunctionBegin;
  ctx->lb = lb;
  ctx->ub = ub;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDApplyProximalMap_L1(TaoPD pd0, TaoPD pd1, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  TaoPDType pd0_type;
  TaoPDType pd1_type;
  PetscBool is_0_l1, is_1_l2;
  TaoPD_L1 *ctx = (TaoPD_L1 *)pd0->data;

  PetscFunctionBegin;
  PetscCall(TaoPDGetType(pd0, &pd0_type));
  PetscCall(TaoPDGetType(pd1, &pd1_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)pd0, TAOPDL1, &is_0_l1));
  PetscCall(PetscObjectTypeCompare((PetscObject)pd1, TAOPDL2, &is_1_l2));
  if (is_1_l2) {
    //Check whats lb ub if unset? TODO
    if (ctx->lb == 0 && ctx->ub == 0) {
      PetscCall(TaoSoftThreshold(y, -lambda, lambda, x));
    } else {
      PetscCall(TaoSoftThreshold(y, ctx->lb, ctx->ub, x));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)pd0), PETSC_ERR_USER, "TaoPDApplyProximalMap_L1 only supports L2 Regularizer types.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Internal Register-Dispatch Version */
PETSC_EXTERN PetscErrorCode TaoPDCreate_L1_Private(TaoPD pd)
{
  TaoPD_L1 *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is 0,0 - i.e., no-op */
  ctx->lb                   = 0.;
  ctx->ub                   = 0.;
  pd->data                  = (void *)ctx;
  pd->ops->applyproximalmap = TaoPDApplyProximalMap_L1;
  pd->ops->destroy          = TaoPDContextDestroy_L1;
  pd->ops->setfromoptions   = TaoPDSetFromOptions_L1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* External Version for User to create L1 in lump-sum fashion TODO add manual `*/
PetscErrorCode TaoPDCreate_L1(MPI_Comm comm, TaoPD *pd, Vec y, PetscReal lb, PetscReal ub)
{
  PetscFunctionBegin;
  PetscCall(TaoPDCreate(comm, pd));
  PetscCall(TaoPDSetType(*pd, TAOPDL1));
  PetscCall(TaoPDSetCentralVector(*pd, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}
