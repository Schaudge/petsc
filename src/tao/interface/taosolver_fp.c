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
  //TODO should i assert step \geq 0, or vm be present?
  PetscCall(VecLockReadPush(in));
  if (tao->ops->computefixedpoint) {
    PetscCall(PetscLogEventBegin(TAO_FixedPointEval, tao, in, out, NULL));
    PetscCall(PetscLogEventEnd(TAO_FixedPointEval, tao, in, out, NULL));
    PetscCallBack("Tao fixed point callback", (*tao->ops->computefixedpoint)(tao, in, out, step, vm, tao->user_fpiP));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoComputeFixedPoint() not available for this solver type.");

  PetscCall(VecLockReadPop(out));
  PetscFunctionReturn(PETSC_SUCCESS);
}
