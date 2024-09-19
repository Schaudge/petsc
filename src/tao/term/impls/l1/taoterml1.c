#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*MC
  TAOTERML1 - A `TaoTerm` that computes $\|x - p\|_1$, for solution $x$ and parameters $p$.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMHALFL2SQUARED`
MC*/
PETSC_INTERN PetscErrorCode TaoTermCreate_L1(TaoTerm term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  term->ops->destroy = TaoTermDestroy_ElementwiseDivergence_Internal;
  PetscFunctionReturn(PETSC_SUCCESS);
}
