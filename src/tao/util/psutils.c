#include <petsc/private/taoimpl.h>
#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/
#include <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <../src/tao/proximal/impls/cv/cv.h> /*I "petsctao.h" I*/

/*@
  TaoPSSetLipschitz - Sets Lipschitz constant of smooth term
  for `TAOFB` or `TAOCV`.

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
- lip - the Lipschitz constant

  Level: intermediate

  Note:
  This method is intended for a usecase where smooth term
  is set via `TaoSetObjective` like routines.

.seealso: [](ch_tao), `Tao`, `TAOFB`, `TAOCV`
@*/
PetscErrorCode TaoPSSetLipschitz(Tao tao, PetscReal lip)
{
  TaoType   type;
  PetscBool isfb,iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, lip, 2);
  PetscCheck(lip > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Lipschitz value has to be greater than zero.");
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));

  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->lip     = lip;
    fb->lip_set = PETSC_TRUE;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->lip     = lip;
    cv->lip_set = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetUseLipApprox - Determine whether to compute initial Lipschitz
  constant approximation or not.

  Input Parameters:
+ tao - the `Tao` context for the `TAOFB` or `TAOCV` solver
- use - Bool to denote whether to compute initial approximiation

  Level: advanced

.seealso: `Tao`, `TAOFB`
@*/
PetscErrorCode TaoPSSetUseLipApprox(Tao tao, PetscBool flag)
{
  TaoType   type;
  PetscBool isfb,iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));

  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->approx_lip = flag;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->approx_lip = flag;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetNonSmoothTerm - Set non-smooth objective term for
  proximal splitting problem - h(x).

  Input Parameters:
+ tao - the `Tao` context for the `TAOFB` and `TAOCV` solver
- dm  - the `DM` context containing non-smooth objective

  Level: advanced

.seealso: `TAOFB`, `TAOCV`, `Tao`, `TaoPSSetNonSmoothTermWithLinearMap()`
@*/
PetscErrorCode TaoPSSetNonSmoothTerm(Tao tao, DM dm)
{
  TaoType   type;
  PetscBool isfb,iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));

  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->proxterm = dm;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->h_prox = dm;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetNonSmoothTermWithLinearMap - Set non-smooth objective term
  with linear mapping for proximal splitting problem - g(Ax).

  Input Parameters:
+ tao  - the `Tao` context for the `TAOCV` solver
. dm   - the `DM` context containing non-smooth objective
. mat  - the linear mapping matrix
- norm - norm of the linear mapping matrix, if avaliable. Set as zero if unknown

  Level: advanced

.seealso: `TAOCV`, `Tao`, `TaoPSSetNonSmoothTerm()`
@*/
PetscErrorCode TaoPSSetNonSmoothTermWithLinearMap(Tao tao, DM dm, Mat mat, PetscReal norm)
{
  TaoType   type;
  PetscBool iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));

  /* Currently this function is only used by TAOCV, but
   * TODO TAODY would use this too */
  if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    if (mat) {
      PetscValidHeaderSpecific(mat, MAT_CLASSID, 3);
      PetscCall(PetscObjectReference((PetscObject)mat));
    }
    PetscValidLogicalCollectiveReal(tao, norm, 4);
    cv->g_prox      = dm;
    cv->g_lmap      = mat;
    cv->g_lmap_norm = norm;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetSmoothTerm - Set smooth objective term for proximal splitting problem.
  This smooth objective - f(x) -  must have gradient term available.

  Input Parameters:
+ tao - the `Tao` context for the `TAOCV`, `TAOFB` solver
- dm  - the `DM` context containing smooth objective

  Level: advanced

.seealso: `TAOFB`, `TAOCV`, `Tao`, `TaoPSSetNonSmoothTerm()`
@*/
PetscErrorCode TaoPSSetSmoothTerm(Tao tao, DM dm)
{
  TaoType   type;
  PetscBool isfb,iscv;
  DMTao   tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCheck(tdm->ops->computeobjectiveandgradient || tdm->ops->computegradient, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "DMTaoSetGradient() has not been called");
  PetscCall(PetscObjectReference((PetscObject)dm));
  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->smoothterm = dm;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->smoothterm = dm;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSUseAcceleration - Determine whether to Nesterov-type
  acceleration strategy for `TAOCV, and `TAOFB`.

  Input Parameters:
+ tao - the `Tao` context for the `TAOCV` or `TAOFB` solver
- use - Bool to denote whether to acceleration

  Level: advanced

.seealso: `Tao`, `TAOFB`, `TAOCV'
@*/
PetscErrorCode TaoPSUseAcceleration(Tao tao, PetscBool flag)
{
  TaoType   type;
  PetscBool isfb,iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->use_accel = PETSC_TRUE;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->use_accel = PETSC_TRUE;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO currently can only use THE hard-coded strategy
/*@
  TaoPSUseAdaptiveStep - Determine whether to use adaptive stepsize
  for `TAOCV, and `TAOFB`.

  Input Parameters:
+ tao - the `Tao` context for the `TAOCV` or `TAOFB` solver
- use - Bool to denote whether to use adaptive stepsize

  Level: advanced

.seealso: `Tao`, `TAOFB`, `TAOCV'
@*/
PetscErrorCode TaoPSUseAdaptiveStep(Tao tao, PetscBool flag)
{
  TaoType   type;
  PetscBool isfb,iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->use_adapt = PETSC_TRUE;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->use_adapt = PETSC_TRUE;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}
