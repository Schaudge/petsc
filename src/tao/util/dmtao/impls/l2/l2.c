#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/
#include <../src/tao/util/dmtao/impls/l2/l2.h>

/* L2 Norm |x|_2^2 DMTao */
//TODO L2 really means L2 norm squared... But i think its fine
static PetscErrorCode DMTaoDestroy_L2(DMTao dm)
{
  DMTao_L2 *l2ctx = (DMTao_L2 *)dm->data;

  PetscFunctionBegin;
  if (l2ctx->workvec2) PetscCall(VecDestroy(&l2ctx->workvec2));
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_L2(DMTao dm, PetscViewer pv)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  DMTao L2"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DMTaoCompute... routines are only if they are Regularizer */
static PetscErrorCode DMTaoComputeObjective_L2(DM dm, Vec X, PetscReal *f, void *ctx)
{
  Vec   y;
  Mat   vm;
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(DMTaoGetCentralVector(dm, &y));
  PetscCall(DMTaoGetVM(dm, &vm));

  DMTao_L2 *l2ctx = (DMTao_L2 *)tdm->data;

  if (!l2ctx->workvec2) PetscCall(VecDuplicate(tdm->workvec, &l2ctx->workvec2));

  if (y) {
    PetscCall(VecWAXPY(tdm->workvec, -1., y, X));
    if (vm) {
      PetscCall(MatMult(vm, tdm->workvec, l2ctx->workvec2));
      PetscCall(VecTDot(tdm->workvec, l2ctx->workvec2, f));
    } else {
      PetscCall(VecNorm(tdm->workvec, NORM_2, f));
    }
  } else {
    if (vm) {
      PetscCall(MatMult(vm, X, tdm->workvec));
      PetscCall(VecTDot(X, tdm->workvec, f));
    } else {
      PetscCall(VecNorm(X, NORM_2, f));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoComputeGradient_L2(DM dm, Vec X, Vec G, void *ctx)
{
  Mat   vm;
  Vec   y;
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(DMTaoGetCentralVector(dm, &y));
  PetscCall(DMTaoGetVM(dm, &vm));

  if (y) {
    if (vm) {
      PetscCall(VecAXPBYPCZ(tdm->workvec, 2, -2, 0, X, y));
      PetscCall(MatMult(vm, tdm->workvec, G));
    } else {
      PetscCall(VecAXPBYPCZ(G, 2, -2, 0, X, y));
    }
  } else {
    if (vm) {
      PetscCall(VecAXPBY(tdm->workvec, 2, 0., X));
      PetscCall(MatMult(vm, tdm->workvec, G));
    } else {
      PetscCall(VecAXPBY(G, 2, 0., X));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoComputeObjectiveAndGradient_L2(DM dm, Vec X, PetscReal *f, Vec G, void *ctx)
{
  Mat   vm;
  Vec   y;
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(DMTaoGetCentralVector(dm, &y));
  PetscCall(DMTaoGetVM(dm, &vm));

  DMTao_L2 *l2ctx = (DMTao_L2 *)tdm->data;

  if (y) {
    if (vm) {
      PetscCall(VecAXPBYPCZ(tdm->workvec, 2, -2, 0, X, y));
      PetscCall(MatMult(vm, tdm->workvec, G));
      PetscCall(VecTDot(tdm->workvec, l2ctx->workvec2, f));
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

static PetscErrorCode DMTaoApplyProximalMap_L2(DM dm0, DM dm1, PetscReal lambda, Vec y, Vec x, PetscBool flg)
{
  PetscBool is_0_l2;
  DMTao     tdm0;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm0, &tdm0));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm0, DMTAOL2, &is_0_l2));
  //TODO
  PetscAssert(is_0_l2, PetscObjectComm((PetscObject)dm0), PETSC_ERR_USER, "L2 Square does not have proximal map.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO Need to implement Hessian Routine */

/*MC
     DMTAOL2 - L2 norm DMTao. This DMTao object represents |x-y|_2^2.

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode DMTaoCreate_L2_Private(DMTao dm)
{
  DMTao_L2 *l2ctx;
  DM        pdm;

  PetscFunctionBegin;
  PetscAssertPointer(dm, 1);
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);

  PetscCall(PetscNew(&l2ctx));
  PetscCall(DMTaoGetParentDM(dm, &pdm));

  pdm->ops->applyproximalmap = DMTaoApplyProximalMap_L2;

  l2ctx->workvec2 = NULL;
  dm->data        = (void *)l2ctx;

  dm->ops->setup                       = NULL;
  dm->ops->destroy                     = DMTaoDestroy_L2;
  dm->ops->view                        = DMTaoView_L2;
  dm->ops->setfromoptions              = NULL;
  dm->ops->reset                       = NULL;
  dm->ops->computeobjective            = DMTaoComputeObjective_L2;
  dm->ops->computegradient             = DMTaoComputeGradient_L2;
  dm->ops->computeobjectiveandgradient = DMTaoComputeObjectiveAndGradient_L2;
  PetscFunctionReturn(PETSC_SUCCESS);
}
