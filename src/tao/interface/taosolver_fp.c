#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@
  TaoComputeFixedPoint - Computes the fixed point characterization of the problem

  Collective

  Input Parameters:
+ tao  - the `Tao` context
- in   - input vector
- step - stepsize
- vm   - variable matrix

  Output Parameter:
. out - output vector

  Level: developer

  Note:
  `TaoComputeFixedPoint()` is typically used within the implementation of the optimization method,
  so most users would not generally call this routine themselves.

.seealso: [](ch_tao), `DMTao`
@*/
PetscErrorCode TaoComputeFixedPoint(Tao tao, Vec in, Vec out, PetscReal step, Mat vm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(in, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(out, VEC_CLASSID, 3);
  PetscCheckSameComm(tao, 1, in, 2);
  PetscCheckSameComm(tao, 1, out, 3);
  PetscCheck(step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize for Fixed Point cannot be negative");
  /* Null case - return the input */
  if (step == 0 && !vm) {
    PetscCall(VecCopy(out, in));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (vm) PetscInfo(tao, "Fixed Point: using VM, ignoring stepsize");
  PetscCall(VecLockReadPush(in));
  if (tao->ops->computefixedpoint) {
    PetscCall(PetscLogEventBegin(TAO_FixedPointEval, tao, in, out, NULL));
    PetscCallBack("Tao fixed point callback", (*tao->ops->computefixedpoint)(tao, in, out, step, vm, tao->user_fpiP));
    PetscCall(PetscLogEventEnd(TAO_FixedPointEval, tao, in, out, NULL));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoComputeFixedPoint() not available for this solver type.");

  PetscCall(VecLockReadPop(in));
  PetscFunctionReturn(PETSC_SUCCESS);
}
