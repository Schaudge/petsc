#include <petsc/private/vecimpl.h>
#include <petscmanagedtype.hpp>

namespace Petsc {

extern PetscErrorCode VecDotAsync(Vec x, Vec y, ManagedScalar &val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);

  PetscCall(PetscLogEventBegin(VEC_Dot, x, y, 0, 0));
  //PetscUseTypeMethod(x, dot, y, val);
  PetscCall(PetscLogEventEnd(VEC_Dot, x, y, 0, 0));
  PetscFunctionReturn(0);
}

extern PetscErrorCode VecNormAsync(Vec x, NormType type, ManagedReal &scal, PetscDeviceContext dctx) {
  PetscBool flg;
  PetscReal rval;

  PetscFunctionBegin;
  /* Cached data? */
  PetscCall(VecNormAvailable(x, type, &flg, &rval));
  if (flg) {
    scal.front(dctx) = rval;
    PetscFunctionReturn(0);
  }

  PetscUseTypeMethod(x, norm, type, scal, dctx);

  if (type != NORM_1_AND_2) {
    if (scal.is_nosync_available(PETSC_MEMTYPE_HOST)) PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[type], scal.front(dctx)));
  }
  PetscFunctionReturn(0);
}

} // namespace Petsc
