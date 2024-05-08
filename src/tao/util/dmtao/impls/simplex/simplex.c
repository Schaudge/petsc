#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/
#include <../src/tao/util/dmtao/impls/simplex/simplex.h>

static PetscErrorCode DMTaoContextDestroy_Simplex(DMTao dm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetFromOptions_Simplex(DMTao dm, PetscOptionItems *PetscOptionsObject)
{
  DMTao_Simplex *ctx = (DMTao_Simplex *)dm->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMTao Simplex options");
  PetscCall(PetscOptionsReal("-dmtao_simplex_tol", "simplex tolerance", "", ctx->tol, &ctx->tol, NULL));
  PetscCall(PetscOptionsReal("-dmtao_simplex_size", "size of simplex", "", ctx->size, &ctx->size, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_Simplex(DMTao tdm, PetscViewer pv)
{
  DMTao_Simplex *ctx = (DMTao_Simplex *)tdm->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  DMTao Simplex"));
    PetscCall(PetscViewerASCIIPrintf(pv, ": tol=%g size=%g ", (double)ctx->tol, (double)ctx->size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscReal Clip_Internal(PetscReal in, PetscReal tmax)
{
  return PetscMax(0, in - tmax);
}

//TODO only supporting comm size one i think?
/* https://arxiv.org/pdf/1101.6081.pdf */
static PetscErrorCode DMTaoApplyProximalMap_Simplex(DMTao tdm0, DMTao tdm1, PetscReal lambda, Vec y, Vec x, PetscBool flg)
{
  PetscBool  is_0_s, is_1_l2  = PETSC_TRUE;
  PetscBool  is_vm_diag, bget = PETSC_FALSE;
  PetscReal *xarray, *yarray, tmax, min, sum, size, cumsum = 0;
  PetscInt   len, i;
  Mat        vm;
  DM         dm1;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm0, DMTAOSIMPLEX, &is_0_s));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm1, DMTAOL2, &is_1_l2));

  PetscCheck(is_0_s, PetscObjectComm((PetscObject)tdm0), PETSC_ERR_USER, "DMTaoApplyProximalMap_Simplex requires first DMTao to be of DMTAOSIMPLEX type.");
  /* dm1 == NULL assumes L2 type */
  if (tdm1) PetscCheck(is_1_l2, PetscObjectComm((PetscObject)tdm1), PETSC_ERR_USER, "DMTaoApplyProximalMap_Simplex only upports L2 regularizer type.");

  DMTao_Simplex *sctx = (DMTao_Simplex *)tdm0->data;

  size = sctx->size;
  PetscCall(VecSum(y, &sum));
  PetscCall(VecMin(y, NULL, &min));
  if (PetscAbsReal(sum - size) < sctx->tol) { PetscFunctionReturn(PETSC_SUCCESS); }

  if (tdm1) {
    PetscCall(DMTaoGetParentDM(tdm1, &dm1));
    PetscCall(DMTaoGetVM(dm1, &vm));
    if (vm) {
      //TODO VM SIMPLEX
      PetscCall(PetscObjectTypeCompare((PetscObject)vm, MATDIAGONAL, &is_vm_diag));
      PetscCheck(is_vm_diag, PetscObjectComm((PetscObject)tdm1), PETSC_ERR_USER, "DMTaoApplyProximalMap_Simplex only supports diagonal VM matrix.");
    }
  }
  PetscCall(VecGetSize(y, &len));
  PetscCall(VecGetArray(y, &yarray));
  PetscCall(PetscSortReal(len, yarray));

  /* Not thinking about parallelism right now... */
  /* Scalar Version. Technically one can do cumulative sum,
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSimplexSetContext - sets the size and tolerance context for simplex DMTao

  Logically Collective

  Input Parameters:
+ dm   - the `DM` containing `DMTAOSIMPLEX`
. size - size of simplex
- tol  - tolerance

  Level: advanced

  Fortran Notes:
  The context can only be an integer or a `PetscObject`

.seealso: `DMTao`, `DMTAOSIMPLEX`
@*/
PetscErrorCode DMTaoSimplexSetContext(DM dm, PetscReal size, PetscReal tol)
{
  PetscBool is_s;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, DMTAOSIMPLEX, &is_s));

  if (!is_s) PetscFunctionReturn(PETSC_SUCCESS);

  DMTao_Simplex *ctx = (DMTao_Simplex *)tdm->data;

  ctx->size = size;
  ctx->tol  = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   DMTAOSIMPLEX - Simplex DMTao object.
Simplex is defined by x_i >= 0, \sum x = size.
If size=1, it is a unit simplex. Proximal mapping is an orthogonal
projection on to this simplex.
TODO support VM

   Options Database Keys:
+      -dmtao_simplex_tol <r> - tolerance
-      -dmtao_simplex_size <r> - size of simplex

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode DMTaoCreate_Simplex_Private(DMTao dm)
{
  DMTao_Simplex *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is unit simplex */
  ctx->size = 1;
#if defined(PETSC_USE_REAL_SINGLE)
  ctx->tol = 1.e-5;
#else
  ctx->tol = 1.e-6;
#endif
  dm->ops->applyproximalmap  = DMTaoApplyProximalMap_Simplex;
  dm->data                   = (void *)ctx;
  dm->ops->setup             = NULL;
  dm->ops->destroy           = DMTaoContextDestroy_Simplex;
  dm->ops->view              = DMTaoView_Simplex;
  dm->ops->setfromoptions    = DMTaoSetFromOptions_Simplex;
  dm->ops->reset             = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
