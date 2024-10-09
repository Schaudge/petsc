
#include <petsc/private/taoimpl.h> /*I "petsctao.h" */

/*@
  TaoTermGradientFD - Approximate the gradient of a `TaoTerm` using finite differences

  Collective

  Input Parameters:
+ tao    - a `TaoTerm`
. x      - a solution vector
- params - (optional) parameters vector (see `TaoTermParametersMode()`)

  Output Paramater:
. g - the computed finit difference approximation to the gradient

  Options Database Keys:
+ -taoterm_fd_delta <delta>       - change in X used to calculate finite differences
- -taoterm_gradient_use_fd <bool> - Use `TaoTermGradientFD()` in `TaoTermGradient()`

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized
  to take advantage of sparsity in the problem.  Although
  not recommended for general use
  in large-scale applications, it can be useful in checking the
  correctness of a user-provided gradient.  Call `TaoTermGradientUseFDPush()` to start using
  this routine in `TaoTermGradient()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermGradientUseFDPush()`,
          `TaoTermGradientUseFDPop()`,
@*/
PetscErrorCode TaoTermGradientFD(TaoTerm term, Vec x, Vec params, Vec g)
{
  Vec          x_perturbed;
  PetscScalar *_g;
  PetscReal    f, f2;
  PetscInt     low, high, N, i;
  PetscReal    h = term->fd_delta;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(x, &x_perturbed));
  PetscCall(VecCopy(x, x_perturbed));
  PetscCall(VecGetSize(x_perturbed, &N));
  PetscCall(VecGetOwnershipRange(x_perturbed, &low, &high));
  PetscCall(VecSetOption(x_perturbed, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(VecGetArray(g, &_g));
  for (i = 0; i < N; i++) {
    PetscCall(VecSetValue(x_perturbed, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    PetscCall(TaoTermObjective(term, x_perturbed, params, &f));
    PetscCall(VecSetValue(x_perturbed, i, 2.0 * h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    PetscCall(TaoTermObjective(term, x_perturbed, params, &f2));
    PetscCall(VecSetValue(x_perturbed, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    if (i >= low && i < high) _g[i - low] = (f2 - f) / (2.0 * h);
  }
  PetscCall(VecRestoreArray(g, &_g));
  PetscCall(VecDestroy(&x_perturbed));
  PetscFunctionReturn(PETSC_SUCCESS);
}
