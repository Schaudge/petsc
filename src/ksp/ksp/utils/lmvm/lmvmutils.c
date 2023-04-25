#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/
#include <petscblaslapack.h>

/*@
  MatLMVMUpdate - Adds (X-Xprev) and (F-Fprev) updates to an LMVM-type matrix.
  The first time the function is called for an LMVM-type matrix, no update is
  applied, but the given X and F vectors are stored for use as Xprev and
  Fprev in the next update.

  If the user has provided another LMVM-type matrix in place of J0, the J0
  matrix is also updated recursively.

  Input Parameters:
+ B - An LMVM-type matrix
. X - Solution vector
- F - Function vector

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMReset()`, `MatLMVMAllocate()`
@*/
PetscErrorCode MatLMVMUpdate(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  if (!lmvm->allocated) {
    PetscCall(MatLMVMAllocate(B, X, F));
  } else {
    VecCheckMatCompatible(B, X, 2, F, 3);
  }
  PetscCall(MatLMVMUpdate(lmvm->J0, X, F));
  PetscCall((*lmvm->ops->update)(B, X, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCreateJ0(Mat B, Mat *J0)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), J0));
  PetscLayout rmap, cmap;
  PetscCall(MatGetLayouts(B, &rmap, &cmap));
  PetscCall(MatSetLayouts(*J0, rmap, cmap));
  VecType vec_type;
  PetscCall(MatGetVecType(B, &vec_type));
  PetscCall(MatSetVecType(*J0, vec_type));
  const char *prefix;
  PetscCall(MatGetOptionsPrefix(B, &prefix));
  PetscCall(MatSetOptionsPrefix(*J0, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*J0, "lmvm_J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCreateJ0KSP(Mat B, KSP *ksp)
{
  PetscFunctionBegin;
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)B), ksp));
  PetscCall(KSPSetOperators(*ksp, lmvm->J0, lmvm->J0));
  const char *prefix;
  PetscCall(MatGetOptionsPrefix(B, &prefix));
  PetscCall(KSPSetOptionsPrefix(*ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(*ksp, "lmvm_J0_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)B, (PetscObject)*ksp, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCreateJ0KSP_ExactInverse(Mat B, KSP *ksp)
{
  PetscFunctionBegin;
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscCall(MatLMVMCreateJ0KSP(B, ksp));
  PetscCall(KSPSetType(*ksp, KSPPREONLY));
  PC pc;
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCMAT));
  PetscCall(PCMatSetApplyOperation(pc, MATOP_SOLVE));
  lmvm->disable_ksp_viewers = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMClearJ0 - Removes all definitions of J0 and reverts to
  an identity matrix (scale = 1.0).

  Input Parameter:
. B - An LMVM-type matrix

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMClearJ0(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&lmvm->J0));
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  PetscCall(MatLMVMCreateJ0(B, &lmvm->J0));
  PetscCall(MatSetType(lmvm->J0, MATCONSTANTDIAGONAL));
  PetscCall(MatZeroEntries(lmvm->J0));
  PetscCall(MatShift(lmvm->J0, 1.0));
  PetscCall(MatLMVMCreateJ0KSP_ExactInverse(B, &lmvm->J0ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0Scale - Allows the user to define a scalar value
  mu such that J0 = mu*I.

  Input Parameters:
+ B     - An LMVM-type matrix
- scale - Scalar value mu that defines the initial Jacobian

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetDiagScale()`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMSetJ0Scale(Mat B, PetscReal scale)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(lmvm->square, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "Scaling is available only for square LMVM matrices");
  PetscBool isdiagonal;

  PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATCONSTANTDIAGONAL, &isdiagonal));
  if (!isdiagonal) { PetscCall(MatLMVMClearJ0(B)); }
  PetscCall(MatZeroEntries(lmvm->J0));
  PetscCall(MatShift(lmvm->J0, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0Diag - Allows the user to define a vector
  V such that J0 = diag(V).

  Input Parameters:
+ B - An LMVM-type matrix
- V - Vector that defines the diagonal of the initial Jacobian: values are copied, V is not referenced

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetScale()`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMSetJ0Diag(Mat B, Vec V)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;
  MPI_Comm  comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(V, VEC_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(lmvm->allocated, comm, PETSC_ERR_ORDER, "Matrix must be allocated before setting diagonal scaling");
  PetscCheck(lmvm->square, comm, PETSC_ERR_SUP, "Diagonal scaling is available only for square LMVM matrices");
  VecCheckSameSize(V, 2, lmvm->Fprev, 3);

  PetscBool isvdiag;

  PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATDIAGONAL, &isvdiag));
  if (!isvdiag) {
    PetscCall(MatLMVMClearJ0(B));
    PetscCall(MatSetType(lmvm->J0, MATDIAGONAL));
  }
  PetscCall(MatDiagonalSet(lmvm->J0, V, INSERT_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetJ0InvDiag(Mat B, Vec *V)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool isvdiag;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATDIAGONAL, &isvdiag));
  if (!isvdiag) {
    PetscCall(MatLMVMClearJ0(B));
    PetscCall(MatSetType(lmvm->J0, MATDIAGONAL));
    PetscCall(MatZeroEntries(lmvm->J0));
    PetscCall(MatShift(lmvm->J0, 1.0));
  }
  PetscCall(MatDiagonalGetInverseDiagonal(lmvm->J0, V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMRestoreJ0InvDiag(Mat B, Vec *V)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatDiagonalRestoreInverseDiagonal(lmvm->J0, V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0 - Allows the user to define the initial
  Jacobian matrix from which the LMVM-type approximation is
  built up. Inverse of this initial Jacobian is applied
  using an internal `KSP` solver, which defaults to `KSPGMRES`.
  This internal `KSP` solver has the "mat_lmvm_" option
  prefix.

  Note that another LMVM-type matrix can be used in place of
  J0, in which case updating the outer LMVM-type matrix will
  also trigger the update for the inner LMVM-type matrix. This
  is useful in cases where a full-memory diagonal approximation
  such as `MATLMVMDIAGBRDN` is used in place of J0.

  Input Parameters:
+ B  - An LMVM-type matrix
- J0 - The initial Jacobian matrix, will be referenced by B.

  Level: advanced

  Note:
  A KSP is created for inverting J0 with prefix "lmvm_J0_" and J0
  is set to both operators in `KSPSetOperators()`.  If you want
  to use a separate preconditioning matrix, use `MatLMVMSetKSP()` directly.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0PC()`, `MatLMVMSetJ0KSP()`
@*/
PetscErrorCode MatLMVMSetJ0(Mat B, Mat J0)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0, MAT_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectReference((PetscObject)J0));
  PetscCall(MatDestroy(&lmvm->J0));
  lmvm->J0 = J0;
  if (lmvm->square) { PetscCall(KSPSetOperators(lmvm->J0ksp, J0, J0)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0PC - Allows the user to define a `PC` object that
  acts as the initial inverse-Jacobian matrix. This `PC` should
  already contain all the operators necessary for its application.
  The LMVM-type matrix only calls `PCApply()` without changing any other
  options.

  Input Parameters:
+ B    - An LMVM-type matrix
- J0pc - `PC` object where `PCApply()` defines an inverse application for J0

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetJ0PC()`
@*/
PetscErrorCode MatLMVMSetJ0PC(Mat B, PC J0pc)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;
  MPI_Comm  comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0pc, PC_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(lmvm->square, comm, PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  Mat J0;
  PetscCall(PCGetOperators(J0pc, &J0, NULL));
  PetscCall(PetscObjectReference((PetscObject)J0));
  PetscCall(MatDestroy(&lmvm->J0));
  lmvm->J0 = J0;
  PetscCall(PetscObjectReference((PetscObject)J0pc));
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  PetscCall(MatLMVMCreateJ0KSP(B, &lmvm->J0ksp));
  PetscCall(KSPSetType(lmvm->J0ksp, KSPPREONLY));
  PetscCall(KSPSetPC(lmvm->J0ksp, J0pc));
  PetscCall(PCDestroy(&J0pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0KSP - Allows the user to provide a pre-configured
  KSP solver for the initial inverse-Jacobian approximation.
  This `KSP` solver should already contain all the operators
  necessary to perform the inversion. The LMVM-type matrix only
  calls `KSPSolve()` without changing any other options.

  Input Parameters:
+ B     - An LMVM-type matrix
- J0ksp - `KSP` solver for the initial inverse-Jacobian application

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetJ0KSP()`
@*/
PetscErrorCode MatLMVMSetJ0KSP(Mat B, KSP J0ksp)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;
  MPI_Comm  comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0ksp, KSP_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(lmvm->square, comm, PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  if (J0ksp != lmvm->J0ksp) lmvm->disable_ksp_viewers = PETSC_FALSE; // If the user supplies a more complicated KSP, don't turn off viewers
  PetscCall(PetscObjectReference((PetscObject)J0ksp));
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  lmvm->J0ksp = J0ksp;
  Mat J0;
  PetscCall(KSPGetOperators(J0ksp, &J0, NULL));
  PetscCall(PetscObjectReference((PetscObject)J0));
  PetscCall(MatDestroy(&lmvm->J0));
  lmvm->J0 = J0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetJ0 - Returns a pointer to the internal J0 matrix.

  Input Parameters:
. B - An LMVM-type matrix

  Output Parameter:
. J0 - `Mat` object for defining the initial Jacobian

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMGetJ0(Mat B, Mat *J0)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *J0 = lmvm->J0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetJ0PC - Returns a pointer to the internal `PC` object
  associated with the initial Jacobian.

  Input Parameter:
. B - An LMVM-type matrix

  Output Parameter:
. J0pc - `PC` object for defining the initial inverse-Jacobian

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0PC()`
@*/
PetscErrorCode MatLMVMGetJ0PC(Mat B, PC *J0pc)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  PetscCall(KSPGetPC(lmvm->J0ksp, J0pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetJ0KSP - Returns a pointer to the internal `KSP` solver
  associated with the initial Jacobian.

  Input Parameter:
. B - An LMVM-type matrix

  Output Parameter:
. J0ksp - `KSP` solver for defining the initial inverse-Jacobian

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0KSP()`
@*/
PetscErrorCode MatLMVMGetJ0KSP(Mat B, KSP *J0ksp)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *J0ksp = lmvm->J0ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMApplyJ0Fwd - Applies an approximation of the forward
  matrix-vector product with the initial Jacobian.

  Input Parameters:
+ B - An LMVM-type matrix
- X - vector to multiply with J0

  Output Parameter:
. Y - resulting vector for the operation

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`, `MatLMVMSetJ0Scale()`, `MatLMVMSetJ0ScaleDiag()`,
          `MatLMVMSetJ0PC()`, `MatLMVMSetJ0KSP()`, `MatLMVMApplyJ0Inv()`
@*/
PetscErrorCode MatLMVMApplyJ0Fwd(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatMult(lmvm->J0, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0HermitianTranspose(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatMultHermitianTranspose(lmvm->J0, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMApplyJ0Inv - Applies some estimation of the initial Jacobian
  inverse to the given vector. The specific form of the application
  depends on whether the user provided a scaling factor, a J0 matrix,
  a J0 `PC`, or a J0 `KSP` object. If no form of the initial Jacobian is
  provided, the function simply does an identity matrix application
  (vector copy).

  Input Parameters:
+ B - An LMVM-type matrix
- X - vector to "multiply" with J0^{-1}

  Output Parameter:
. Y - resulting vector for the operation

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`, `MatLMVMSetJ0Scale()`, `MatLMVMSetJ0ScaleDiag()`,
          `MatLMVMSetJ0PC()`, `MatLMVMSetJ0KSP()`, `MatLMVMApplyJ0Fwd()`
@*/
PetscErrorCode MatLMVMApplyJ0Inv(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPushGetViewerOff(PETSC_TRUE));
  PetscCall(KSPSolve(lmvm->J0ksp, X, Y));
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPopGetViewerOff());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0InvTranspose(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPushGetViewerOff(PETSC_TRUE));
  PetscCall(KSPSolveTranspose(lmvm->J0ksp, X, Y));
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPopGetViewerOff());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0InvHermitianTranspose(Mat B, Vec X, Vec Y)
{
  PetscFunctionBegin;
  if (!PetscDefined(USE_COMPLEX)) {
    PetscCall(MatLMVMApplyJ0InvTranspose(B, X, Y));
  } else {
    Vec X_conj;

    PetscCall(VecDuplicate(X, &X_conj));
    PetscCall(VecCopy(X, X_conj));
    PetscCall(VecConjugate(X_conj));
    PetscCall(MatLMVMApplyJ0InvTranspose(B, X_conj, Y));
    PetscCall(VecConjugate(Y));
    PetscCall(VecDestroy(&X_conj));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMIsAllocated - Returns a boolean flag that shows whether
  the necessary data structures for the underlying matrix is allocated.

  Input Parameter:
. B - An LMVM-type matrix

  Output Parameter:
. flg - `PETSC_TRUE` if allocated, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMAllocate()`, `MatLMVMReset()`
@*/
PetscErrorCode MatLMVMIsAllocated(Mat B, PetscBool *flg)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *flg = PETSC_FALSE;
  if (lmvm->allocated && B->preallocated && B->assembled) *flg = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMAllocate - Produces all necessary common memory for
  LMVM approximations based on the solution and function vectors
  provided. If `MatSetSizes()` and `MatSetUp()` have not been called
  before `MatLMVMAllocate()`, the allocation will read sizes from
  the provided vectors and update the matrix.

  Input Parameters:
+ B - An LMVM-type matrix
. X - Solution vector
- F - Function vector

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMReset()`, `MatLMVMUpdate()`
@*/
PetscErrorCode MatLMVMAllocate(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;
  VecType   vtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetType(X, &vtype));
  PetscCall(MatSetVecType(B, vtype));
  PetscCall((*lmvm->ops->allocate)(B, X, F));
  PetscCall(MatLMVMAllocate(lmvm->J0, X, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMResetShift - Zero the shift factor.

  Input Parameter:
. B - An LMVM-type matrix

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMAllocate()`, `MatLMVMUpdate()`
@*/
PetscErrorCode MatLMVMResetShift(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm->shift = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMReset - Flushes all of the accumulated updates out of
  the LMVM approximation. In practice, this will not actually
  destroy the data associated with the updates. It simply resets
  counters, which leads to existing data being overwritten, and
  `MatSolve()` being applied as if there are no updates. A boolean
  flag is available to force destruction of the update vectors.

  If the user has provided another LMVM matrix as J0, the J0
  matrix is also reset in this function.

  Input Parameters:
+ B           - An LMVM-type matrix
- destructive - flag for enabling destruction of data structures

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMAllocate()`, `MatLMVMUpdate()`
@*/
PetscErrorCode MatLMVMReset(Mat B, PetscBool destructive)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatLMVMReset(lmvm->J0, destructive));
  PetscCall((*lmvm->ops->reset)(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetHistorySize - Set the number of past iterates to be
  stored for the construction of the limited-memory QN update.

  Input Parameters:
+ B         - An LMVM-type matrix
- hist_size - number of past iterates (default 5)

  Options Database Key:
. -mat_lmvm_hist_size <m> - set number of past iterates

  Level: beginner

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetUpdateCount()`
@*/
PetscErrorCode MatLMVMSetHistorySize(Mat B, PetscInt hist_size)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;
  Vec       X, F;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(hist_size >= 0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "QN history size must be a non-negative integer.");
  if (lmvm->allocated && lmvm->m != hist_size) {
    PetscCall(VecDuplicate(lmvm->Xprev, &X));
    PetscCall(VecDuplicate(lmvm->Fprev, &F));
    PetscCall(MatLMVMReset(B, PETSC_TRUE));
    lmvm->m = hist_size;
    PetscCall(MatLMVMAllocate(B, X, F));
    PetscCall(VecDestroy(&X));
    PetscCall(VecDestroy(&F));
  }
  lmvm->m = hist_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetUpdateCount - Returns the number of accepted updates.
  This number may be greater than the total number of update vectors
  stored in the matrix. The counters are reset when `MatLMVMReset()`
  is called.

  Input Parameter:
. B - An LMVM-type matrix

  Output Parameter:
. nupdates - number of accepted updates

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetRejectCount()`, `MatLMVMReset()`
@*/
PetscErrorCode MatLMVMGetUpdateCount(Mat B, PetscInt *nupdates)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *nupdates = lmvm->nupdates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetRejectCount - Returns the number of rejected updates.
  The counters are reset when `MatLMVMReset()` is called.

  Input Parameter:
. B - An LMVM-type matrix

  Output Parameter:
. nrejects - number of rejected updates

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetRejectCount()`, `MatLMVMReset()`
@*/
PetscErrorCode MatLMVMGetRejectCount(Mat B, PetscInt *nrejects)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *nrejects = lmvm->nrejects;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMUpdateOpVecs(Mat B, LMBasis X, LMBasis OpX, PetscErrorCode (*op)(Mat, Vec, Vec))
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscObjectId J0_id;
  PetscCall(PetscObjectGetId((PetscObject)lmvm->J0, &J0_id));
  PetscObjectState J0_state;
  PetscCall(PetscObjectStateGet((PetscObject)lmvm->J0, &J0_state));
  PetscInt oldest, next;
  PetscCall(LMBasisGetRange(X, &oldest, &next));
  if (OpX->operator_id != J0_id || OpX->operator_state != J0_state) {
    // invalidate OpX
    OpX->k              = oldest;
    OpX->operator_id    = J0_id;
    OpX->operator_state = J0_state;
  }
  OpX->k = PetscMax(OpX->k, oldest);
  for (PetscInt i = OpX->k; i < next; i++) {
    Vec x_i, op_x_i;

    PetscCall(LMBasisGetVec(X, i, PETSC_MEMORY_ACCESS_READ, &x_i));
    PetscCall(LMBasisGetNextVec(OpX, &op_x_i));
    PetscCall(op(B, x_i, op_x_i));
    PetscCall(LMBasisRestoreNextVec(OpX, &op_x_i));
    PetscCall(LMBasisRestoreVec(X, i, PETSC_MEMORY_ACCESS_READ, &x_i));
  }
  PetscAssert(OpX->k == X->k && OpX->operator_id == J0_id && OpX->operator_state == J0_state, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Invalid state for operator-updated LMBasis");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMUpdateOpDiffVecs(Mat B, LMBasis Y, LMBasis OpX, LMBasis YmOpX)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscObjectId J0_id;
  PetscCall(PetscObjectGetId((PetscObject)lmvm->J0, &J0_id));
  PetscObjectState J0_state;
  PetscCall(PetscObjectStateGet((PetscObject)lmvm->J0, &J0_state));
  PetscInt oldest, next;
  PetscAssert(Y->m == OpX->m, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Incompatible Y and OpX in MatLMVMUpdateOpDiffVecs()");
  PetscAssert(Y->k == OpX->k && OpX->operator_id == J0_id && OpX->operator_state == J0_state, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Stale OpX in MatLMVMUpdateOpDiffVecs()");
  PetscCall(LMBasisGetRange(Y, &oldest, &next));
  if (YmOpX->operator_id != J0_id || YmOpX->operator_state != J0_state) {
    // invalidate OpX
    YmOpX->k              = oldest;
    YmOpX->operator_id    = J0_id;
    YmOpX->operator_state = J0_state;
  }
  YmOpX->k       = PetscMax(YmOpX->k, oldest);
  PetscInt start = YmOpX->k;
  if (next - start == Y->m) { // full matrix AXPY
    PetscCall(MatCopy(Y->vecs, YmOpX->vecs, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(YmOpX->vecs, -1.0, OpX->vecs, SAME_NONZERO_PATTERN));
    YmOpX->k = Y->k;
  } else {
    for (PetscInt i = start; i < next; i++) {
      Vec y_i, op_x_i, y_m_op_x_i;

      PetscCall(LMBasisGetVec(Y, i, PETSC_MEMORY_ACCESS_READ, &y_i));
      PetscCall(LMBasisGetVec(OpX, i, PETSC_MEMORY_ACCESS_READ, &op_x_i));
      PetscCall(LMBasisGetNextVec(YmOpX, &y_m_op_x_i));
      PetscCall(VecAXPBYPCZ(y_m_op_x_i, 1.0, -1.0, 0.0, y_i, op_x_i));
      PetscCall(LMBasisRestoreNextVec(YmOpX, &y_m_op_x_i));
      PetscCall(LMBasisRestoreVec(OpX, i, PETSC_MEMORY_ACCESS_READ, &op_x_i));
      PetscCall(LMBasisRestoreVec(Y, i, PETSC_MEMORY_ACCESS_READ, &y_i));
    }
  }
  PetscAssert(YmOpX->k == Y->k && YmOpX->operator_id == J0_id && YmOpX->operator_state == J0_state, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Invalid state for operator-updated LMBasis");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedBasis(Mat B, MatLMVMBasisType type, LMBasis *basis_p)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  LMBasis   basis;

  PetscFunctionBegin;
  if (!lmvm->basis[type]) PetscCall(LMBasisCreate(MatLMVMBasisSizeOf(type) == LMBASIS_S ? lmvm->Xprev : lmvm->Fprev, lmvm->m, &lmvm->basis[type]));
  basis = lmvm->basis[type];
  switch (type) {
  case LMBASIS_B0S:
    PetscCall(MatLMVMUpdateOpVecs(B, lmvm->basis[LMBASIS_S], basis, MatLMVMApplyJ0Fwd));
    break;
  case LMBASIS_H0Y:
    PetscCall(MatLMVMUpdateOpVecs(B, lmvm->basis[LMBASIS_Y], basis, MatLMVMApplyJ0Inv));
    break;
  case LMBASIS_S_MINUS_H0Y: {
    LMBasis H0Y;
    PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_H0Y, &H0Y));
    PetscCall(MatLMVMUpdateOpDiffVecs(B, lmvm->basis[LMBASIS_S], H0Y, basis));
  } break;
  case LMBASIS_Y_MINUS_B0S: {
    LMBasis B0S;
    PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_B0S, &B0S));
    PetscCall(MatLMVMUpdateOpDiffVecs(B, lmvm->basis[LMBASIS_Y], B0S, basis));
  } break;
  default:
    break;
  }
  *basis_p = basis;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetVecsRead_Internal(Mat B, PetscInt idx, ...)
{
  PetscFunctionBegin;
  va_list ap;
  va_start(ap, idx);
  while (1) {
    MatLMVMBasisType type = (MatLMVMBasisType)va_arg(ap, int);

    if (type == LMBASIS_END) break;

    Vec *vec = va_arg(ap, Vec *);

    LMBasis basis;
    PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
    PetscCall(LMBasisGetVec(basis, idx, PETSC_MEMORY_ACCESS_READ, vec));
  }
  va_end(ap);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMRestoreVecsRead_Internal(Mat B, PetscInt idx, ...)
{
  PetscFunctionBegin;

  va_list ap;
  va_start(ap, idx);
  while (1) {
    MatLMVMBasisType type = (MatLMVMBasisType)va_arg(ap, int);

    if (type == LMBASIS_END) break;

    Vec    *vec = va_arg(ap, Vec *);
    LMBasis basis;
    PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
    PetscCall(LMBasisRestoreVec(basis, idx, PETSC_MEMORY_ACCESS_READ, vec));
  }
  va_end(ap);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetRange(Mat B, PetscInt *oldest, PetscInt *next)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(LMBasisGetRange(lmvm->basis[LMBASIS_S], oldest, next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisGetWorkRow(Mat B, MatLMVMBasisType type, PetscScalar **array_p)
{
  LMBasis basis;

  PetscFunctionBegin;
  type = MatLMVMBasisSizeOf(type);
  PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
  PetscCall(LMBasisGetWorkRow(basis, array_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisRestoreWorkRow(Mat B, MatLMVMBasisType type, PetscScalar **array_p)
{
  LMBasis basis;

  PetscFunctionBegin;
  type = MatLMVMBasisSizeOf(type);
  PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
  PetscCall(LMBasisRestoreWorkRow(basis, array_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisGetWorkVec(Mat B, MatLMVMBasisType type, Vec *vec_p)
{
  LMBasis basis;

  PetscFunctionBegin;
  type = MatLMVMBasisSizeOf(type);
  PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
  PetscCall(LMBasisGetWorkVec(basis, vec_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisRestoreWorkVec(Mat B, MatLMVMBasisType type, Vec *vec_p)
{
  LMBasis basis;

  PetscFunctionBegin;
  type = MatLMVMBasisSizeOf(type);
  PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
  PetscCall(LMBasisRestoreWorkVec(basis, vec_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMApplyOpThenVecs(PetscScalar alpha, Mat B, PetscInt oldest, PetscInt next, MatLMVMBasisType type_S, PetscErrorCode (*op)(Mat, Vec, Vec), Vec x, PetscReal beta, PetscScalar y[])
{
  LMBasis S;
  Vec     B0H_v;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, type_S, &S));
  PetscCall(LMBasisGetWorkVec(S, &B0H_v));
  PetscCall(op(B, x, B0H_v));
  PetscCall(LMBasisGEMVH(alpha, S, oldest, next, B0H_v, beta, y));
  PetscCall(LMBasisRestoreWorkVec(S, &B0H_v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMApplyVecsThenOp(PetscScalar alpha, Mat B, PetscInt oldest, PetscInt next, MatLMVMBasisType type_S, MatLMVMBasisType type_Y, PetscErrorCode (*op)(Mat, Vec, Vec), PetscScalar x[], PetscReal beta, Vec y)
{
  LMBasis S, Y;
  Vec     S_x;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, type_S, &S));
  PetscCall(MatLMVMGetUpdatedBasis(B, type_Y, &Y));
  PetscCall(LMBasisGetWorkVec(S, &S_x));
  PetscCall(LMBasisGEMV(alpha, S, oldest, next, x, 0.0, S_x));
  if (beta == 0.0) {
    PetscCall(op(B, S_x, y));
  } else {
    Vec B0S_x;
    PetscCall(LMBasisGetWorkVec(Y, &B0S_x));
    PetscCall(op(B, S_x, B0S_x));
    PetscCall(VecAYPX(y, beta, B0S_x));
    PetscCall(LMBasisRestoreWorkVec(Y, &B0S_x));
  }
  PetscCall(LMBasisRestoreWorkVec(S, &S_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisGEMVH(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec v, Vec op_v, PetscScalar beta, PetscScalar array[])
{
  Mat_LMVM *lmvm              = (Mat_LMVM *)B->data;
  PetscBool cache_J0_products = lmvm->do_not_cache_J0_products ? PETSC_FALSE : PETSC_TRUE;
  LMBasis   basis;

  PetscFunctionBegin;
  if (cache_J0_products || type == LMBASIS_S || type == LMBASIS_Y) {
    PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
    PetscCall(LMBasisGEMVH(alpha, basis, oldest, next, v, beta, array));
  } else {
    switch (type) {
    case LMBASIS_B0S:
      if (op_v) {
        PetscCall(MatLMVMBasisGEMVH(B, LMBASIS_S, oldest, next, alpha, op_v, NULL, beta, array));
      } else {
        PetscCall(MatLMVMApplyOpThenVecs(alpha, B, oldest, next, LMBASIS_S, MatLMVMApplyJ0HermitianTranspose, v, beta, array));
      }
      break;
    case LMBASIS_H0Y:
      if (op_v) {
        PetscCall(MatLMVMBasisGEMVH(B, LMBASIS_Y, oldest, next, alpha, op_v, NULL, beta, array));
      } else {
        PetscCall(MatLMVMApplyOpThenVecs(alpha, B, oldest, next, LMBASIS_Y, MatLMVMApplyJ0InvHermitianTranspose, v, beta, array));
      }
      break;
    case LMBASIS_Y_MINUS_B0S:
      if (op_v) {
        PetscCall(MatLMVMBasisGEMVH(B, LMBASIS_Y, oldest, next, alpha, v, NULL, beta, array));
        PetscCall(MatLMVMBasisGEMVH(B, LMBASIS_S, oldest, next, -alpha, op_v, NULL, 1.0, array));
      } else {
        PetscCall(MatLMVMApplyOpThenVecs(-alpha, B, oldest, next, LMBASIS_S, MatLMVMApplyJ0HermitianTranspose, v, beta, array));
        PetscCall(LMBasisGEMVH(alpha, lmvm->basis[LMBASIS_Y], oldest, next, v, 1.0, array));
      }
      break;
    case LMBASIS_S_MINUS_H0Y:
      if (op_v) {
        PetscCall(MatLMVMBasisGEMVH(B, LMBASIS_S, oldest, next, alpha, v, NULL, beta, array));
        PetscCall(MatLMVMBasisGEMVH(B, LMBASIS_Y, oldest, next, -alpha, op_v, NULL, 1.0, array));
      } else {
        PetscCall(MatLMVMApplyOpThenVecs(-alpha, B, oldest, next, LMBASIS_Y, MatLMVMApplyJ0InvHermitianTranspose, v, beta, array));
        PetscCall(LMBasisGEMVH(alpha, lmvm->basis[LMBASIS_S], oldest, next, v, 1.0, array));
      }
      break;
    default:
      PetscUnreachable();
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisMultHermitianTranspose(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, Vec v, Vec op_v, PetscScalar array[])
{
  PetscFunctionBegin;
  PetscCall(MatLMVMBasisGEMVH(B, type, oldest, next, 1.0, v, op_v, 0.0, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// x must come from MatLMVMGetRowWork()
PETSC_INTERN PetscErrorCode MatLMVMBasisGEMV(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, PetscScalar alpha, PetscScalar x[], PetscScalar beta, Vec y)
{
  Mat_LMVM *lmvm              = (Mat_LMVM *)B->data;
  PetscBool cache_J0_products = lmvm->do_not_cache_J0_products ? PETSC_FALSE : PETSC_TRUE;
  LMBasis   basis;

  PetscFunctionBegin;
  if (cache_J0_products || type == LMBASIS_S || type == LMBASIS_Y) {
    PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis));
    PetscCall(LMBasisGEMV(alpha, basis, oldest, next, x, beta, y));
  } else {
    switch (type) {
    case LMBASIS_B0S:
      PetscCall(MatLMVMApplyVecsThenOp(alpha, B, oldest, next, LMBASIS_S, LMBASIS_Y, MatLMVMApplyJ0Fwd, x, beta, y));
      break;
    case LMBASIS_H0Y:
      PetscCall(MatLMVMApplyVecsThenOp(alpha, B, oldest, next, LMBASIS_Y, LMBASIS_S, MatLMVMApplyJ0Inv, x, beta, y));
      break;
    case LMBASIS_Y_MINUS_B0S:
      PetscCall(LMBasisGEMV(alpha, lmvm->basis[LMBASIS_Y], oldest, next, x, beta, y));
      PetscCall(MatLMVMApplyVecsThenOp(-alpha, B, oldest, next, LMBASIS_S, LMBASIS_Y, MatLMVMApplyJ0Fwd, x, 1.0, y));
      break;
    case LMBASIS_S_MINUS_H0Y:
      PetscCall(LMBasisGEMV(alpha, lmvm->basis[LMBASIS_S], oldest, next, x, beta, y));
      PetscCall(MatLMVMApplyVecsThenOp(-alpha, B, oldest, next, LMBASIS_Y, LMBASIS_S, MatLMVMApplyJ0Inv, x, 1.0, y));
      break;
    default:
      PetscUnreachable();
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisMultAdd(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, PetscScalar x[], Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatLMVMBasisGEMV(B, type, oldest, next, 1.0, x, 1.0, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGramianSolve(Mat B, PetscInt oldest, PetscInt next, MatLMVMBasisType X, MatLMVMBasisType Y, LMSolveType solve_type, PetscScalar b[], PetscBool hermitian_transpose)
{
  LMBlockType block_type = LMBlockTypeFromSolveType(solve_type);
  LMGramian   lmwd;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedGramian(B, X, Y, block_type, &lmwd));
  PetscCall(LMGramianSolve(lmwd, oldest, next, solve_type, b, hermitian_transpose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGramianUpdate(Mat B, MatLMVMBasisType type_X, MatLMVMBasisType type_Y, LMBlockType block_type)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  LMBasis   X, Y;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, type_X, &X));
  PetscCall(MatLMVMGetUpdatedBasis(B, type_Y, &Y));
  if (!lmvm->gramian[type_X][type_Y]) PetscCall(LMGramianCreate(lmvm->m, &lmvm->gramian[type_X][type_Y]));

  PetscCall(LMGramianUpdateBlock(lmvm->gramian[type_X][type_Y], X, Y, block_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedGramian(Mat B, MatLMVMBasisType type_X, MatLMVMBasisType type_Y, LMBlockType block_type, LMGramian *lmwd)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatLMVMGramianUpdate(B, type_X, type_Y, block_type));
  *lmwd = lmvm->gramian[type_X][type_Y];
  PetscFunctionReturn(PETSC_SUCCESS);
}
