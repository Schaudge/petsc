#include <petsc/private/taopdimpl.h>
#include <../src/tao/pd/impls/simplex/simplex.h>
#include <../src/tao/pd/impls/l2/l2.h>

static PetscErrorCode TaoPDContextDestroy_Simplex(TaoPD pd)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(pd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPDSetFromOptions_Simplex(TaoPD pd, PetscOptionItems *PetscOptionsObject)
{
  TaoPD_Simplex *ctx = (TaoPD_Simplex *)pd->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoPD Simplex options");
  PetscCall(PetscOptionsReal("-tao_pd_simplex_tol", "simplex tolerance", "", ctx->tol, &ctx->tol, NULL));
  PetscCall(PetscOptionsReal("-tao_pd_simplex_size", "size of simplex", "", ctx->size, &ctx->size, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoPDSimplexSetContext(TaoPD pd, PetscReal size, PetscReal tol)
{
  TaoPD_Simplex *ctx = (TaoPD_Simplex *)pd->data;
  PetscBool      is_s;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(pd->type, TAOPDSIMPLEX, &is_s));
  if (!is_s) PetscFunctionReturn(PETSC_SUCCESS);
  ctx->size = size;
  ctx->tol  = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscReal Clip_Internal(PetscReal in, PetscReal tmax)
{
  return PetscMax(0, in - tmax);
}

/* https://arxiv.org/pdf/1101.6081.pdf */
static PetscErrorCode TaoPDApplyProximalMap_Simplex(TaoPD pd0, TaoPD pd1, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  TaoPD_Simplex *sctx  = (TaoPD_Simplex *)pd0->data;
  TaoPD_L2      *l2ctx = (TaoPD_L2 *)pd1->data;
  TaoPDType      pd0_type;
  TaoPDType      pd1_type;
  PetscBool      is_0_s, is_1_l2, is_vm_diag, bget = PETSC_FALSE;
  PetscReal     *xarray, *yarray, tmax, min, sum, size, cumsum = 0;
  PetscInt       len, i;
  Mat            vm;

  PetscFunctionBegin;
  PetscCall(TaoPDGetType(pd0, &pd0_type));
  PetscCall(TaoPDGetType(pd1, &pd1_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)pd0, TAOPDSIMPLEX, &is_0_s));
  PetscCall(PetscObjectTypeCompare((PetscObject)pd1, TAOPDL2, &is_1_l2));
  PetscCheck(is_1_l2, PetscObjectComm((PetscObject)pd0), PETSC_ERR_USER, "TaoPDApplyProximalMap_Simplex only upports L2 regularizer type.");

  size = sctx->size;
  PetscCall(VecSum(y, &sum));
  PetscCall(VecMin(y, NULL, &min));
  if (PetscAbsReal(sum - size) < sctx->tol) { PetscFunctionReturn(PETSC_SUCCESS); }

  vm = l2ctx->vm;
  //TODO how to handle VM flag?
  if (vm) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vm, MATDIAGONAL, &is_vm_diag));
    PetscCheck(is_vm_diag, PetscObjectComm((PetscObject)pd0), PETSC_ERR_USER, "TaoPDApplyProximalMap_Simplex only supports diagonal VM matrix.");
  }
  PetscCall(VecGetSize(y, &len));
  PetscCall(VecGetArray(y, &yarray));
  PetscCall(PetscSortReal(len, yarray));

  /* Not thinking about parallelism right now... */
  /* Scalar Version. Technically one can du cumulative sum,
   * but VecCumSum/PrefixSum isn't there yet. */
  for (i = len - 1; i >= 0; i--) {
    cumsum += yarray[i];
    tmax = (cumsum - size) / (len - i);
    if (tmax >= yarray[i - 1]) {
      bget = PETSC_TRUE;
      break;
    }
  }
  if (!bget) { tmax = (cumsum - size) / len; }

  /* Ideally, VecShift(y, -tmax), and set all neg to 0 in two kernel calls... */
  PetscCall(VecGetArray(x, &xarray));
  for (i = 0; i < len; i++) { xarray[i] = Clip_Internal(yarray[i], tmax); }
  PetscCall(VecRestoreArray(y, &yarray));
  PetscCall(VecRestoreArray(x, &xarray));
  //TODO support diag vm metric
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPDCreate_Simplex_Private(TaoPD pd)
{
  TaoPD_Simplex *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is unit simplex */
  ctx->size = 1;
#if defined(PETSC_USE_REAL_SINGLE)
  ctx->tol = 1.e-5;
#else
  ctx->tol = 1.e-5;
#endif
  pd->data                  = (void *)ctx;
  pd->ops->applyproximalmap = TaoPDApplyProximalMap_Simplex;
  pd->ops->destroy          = TaoPDContextDestroy_Simplex;
  pd->ops->setfromoptions   = TaoPDSetFromOptions_Simplex;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoPDCreate_Simplex(MPI_Comm comm, TaoPD *pd, Vec y, PetscReal size, PetscReal tol)
{
  PetscFunctionBegin;
  PetscCall(TaoPDCreate(comm, pd));
  PetscCall(TaoPDSetType(*pd, TAOPDSIMPLEX));
  PetscCall(TaoPDSetCentralVector(*pd, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}
