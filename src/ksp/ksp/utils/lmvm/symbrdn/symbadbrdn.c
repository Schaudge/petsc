#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_LMVMSymBadBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscScalar  yjtqi, sjtyi, wtyi, ytx, stf, wtf, ytq;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi_scalar == 0.0) {
    PetscCall(MatSolve_LMVMBFGS(B, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (lsb->phi_scalar == 1.0) {
    PetscCall(MatSolve_LMVMDFP(B, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (PetscInt i = 0; i < next - oldest; ++i) {
      Vec y_i;
      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
      PetscCall(MatSymBrdnApplyJ0Inv(B, y_i, lsb->Q[i]));
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar yjtsj;
        Vec         s_j, y_j;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + j, &yjtsj));
        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_S, &s_j, LMBASIS_Y, &y_j));
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(y_j, lsb->Q[i], &yjtqi));
        PetscCall(VecDotBegin(s_j, y_i, &sjtyi));
        PetscCall(VecDotEnd(y_j, lsb->Q[i], &yjtqi));
        PetscCall(VecDotEnd(s_j, y_i, &sjtyi));
        /* Compute the pure DFP component of the inverse application*/
        PetscCall(VecAXPBYPCZ(lsb->Q[i], -PetscRealPart(yjtqi) / lsb->ytq[j], PetscRealPart(sjtyi) / yjtsj, 1.0, lsb->Q[j], s_j));
        /* Tack on the convexly scaled extras to the inverse application*/
        if (lsb->psi[j] > 0.0) {
          PetscCall(VecAXPBYPCZ(lsb->work, 1.0 / yjtsj, -1.0 / lsb->ytq[j], 0.0, s_j, lsb->Q[j]));
          PetscCall(VecDot(lsb->work, y_i, &wtyi));
          PetscCall(VecAXPY(lsb->Q[i], lsb->phi_scalar * lsb->ytq[j] * PetscRealPart(wtyi), lsb->work));
        }
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_S, &s_j, LMBASIS_Y, &y_j));
      }
      PetscCall(VecDot(y_i, lsb->Q[i], &ytq));
      lsb->ytq[i] = PetscRealPart(ytq);
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
    }
    lsb->needQ = PETSC_FALSE;
  }

  /* Start the outer iterations for ((B^{-1}) * dX) */
  PetscCall(MatSymBrdnApplyJ0Inv(B, F, dX));
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscScalar yitsi;
    Vec         s_i, y_i;

    PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &yitsi));
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
    /* Compute the necessary dot products -- store yTs and yTp for inner iterations later */
    PetscCall(VecDotBegin(y_i, dX, &ytx));
    PetscCall(VecDotBegin(s_i, F, &stf));
    PetscCall(VecDotEnd(y_i, dX, &ytx));
    PetscCall(VecDotEnd(s_i, F, &stf));
    /* Compute the pure DFP component */
    PetscCall(VecAXPBYPCZ(dX, -PetscRealPart(ytx) / lsb->ytq[i], PetscRealPart(stf) / yitsi, 1.0, lsb->Q[i], s_i));
    /* Tack on the convexly scaled extras */
    PetscCall(VecAXPBYPCZ(lsb->work, 1.0 / yitsi, -1.0 / lsb->ytq[i], 0.0, s_i, lsb->Q[i]));
    PetscCall(VecDot(lsb->work, F, &wtf));
    PetscCall(VecAXPY(dX, lsb->phi_scalar * lsb->ytq[i] * PetscRealPart(wtf), lsb->work));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMSymBadBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscReal    numer;
  PetscScalar  sjtpi, sjtyi, yjtsi, yjtqi, wtsi, wtyi, stz, ytx, ytq, wtx, stp;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi_scalar == 0.0) {
    PetscCall(MatMult_LMVMBFGS(B, X, Z));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (lsb->phi_scalar == 1.0) {
    PetscCall(MatMult_LMVMDFP(B, X, Z));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (PetscInt i = 0; i < next - oldest; ++i) {
      Vec y_i;
      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
      PetscCall(MatSymBrdnApplyJ0Inv(B, y_i, lsb->Q[i]));
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar yjtsj;
        Vec         s_j, y_j;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + j, &yjtsj));
        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_S, &s_j, LMBASIS_Y, &y_j));
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(y_j, lsb->Q[i], &yjtqi));
        PetscCall(VecDotBegin(s_j, y_i, &sjtyi));
        PetscCall(VecDotEnd(y_j, lsb->Q[i], &yjtqi));
        PetscCall(VecDotEnd(s_j, y_i, &sjtyi));
        /* Compute the pure DFP component of the inverse application*/
        PetscCall(VecAXPBYPCZ(lsb->Q[i], -PetscRealPart(yjtqi) / lsb->ytq[j], PetscRealPart(sjtyi) / yjtsj, 1.0, lsb->Q[j], s_j));
        /* Tack on the convexly scaled extras to the inverse application*/
        if (lsb->psi[j] > 0.0) {
          PetscCall(VecAXPBYPCZ(lsb->work, 1.0 / yjtsj, -1.0 / lsb->ytq[j], 0.0, s_j, lsb->Q[j]));
          PetscCall(VecDot(lsb->work, y_i, &wtyi));
          PetscCall(VecAXPY(lsb->Q[i], lsb->phi_scalar * lsb->ytq[j] * PetscRealPart(wtyi), lsb->work));
        }
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_S, &s_j, LMBASIS_Y, &y_j));
      }
      PetscCall(VecDot(y_i, lsb->Q[i], &ytq));
      lsb->ytq[i] = PetscRealPart(ytq);
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
    }
    lsb->needQ = PETSC_FALSE;
  }
  if (lsb->needP) {
    /* Start the loop for (P[k] = (B_k) * S[k]) */
    for (PetscInt i = 0; i < next - oldest; ++i) {
      Vec s_i;
      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i));
      PetscCall(MatSymBrdnApplyJ0Fwd(B, s_i, lsb->P[i]));
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar yjtsj;
        Vec         s_j, y_j;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + j, &yjtsj));
        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_S, &s_j, LMBASIS_Y, &y_j));
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(s_j, lsb->P[i], &sjtpi));
        PetscCall(VecDotBegin(y_j, s_i, &yjtsi));
        PetscCall(VecDotEnd(s_j, lsb->P[i], &sjtpi));
        PetscCall(VecDotEnd(y_j, s_i, &yjtsi));
        /* Compute the pure BFGS component of the forward product */
        PetscCall(VecAXPBYPCZ(lsb->P[i], -PetscRealPart(sjtpi) / lsb->stp[j], PetscRealPart(yjtsi) / yjtsj, 1.0, lsb->P[j], y_j));
        /* Tack on the convexly scaled extras to the forward product */
        if (lsb->phi_scalar > 0.0) {
          PetscCall(VecAXPBYPCZ(lsb->work, 1.0 / yjtsj, -1.0 / lsb->stp[j], 0.0, y_j, lsb->P[j]));
          PetscCall(VecDot(lsb->work, s_i, &wtsi));
          PetscCall(VecAXPY(lsb->P[i], lsb->psi[j] * lsb->stp[j] * PetscRealPart(wtsi), lsb->work));
        }
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_S, &s_j, LMBASIS_Y, &y_j));
      }
      PetscCall(VecDot(s_i, lsb->P[i], &stp));
      lsb->stp[i] = PetscRealPart(stp);
      if (lsb->phi_scalar == 1.0) {
        lsb->psi[i] = 0.0;
      } else if (lsb->phi_scalar == 0.0) {
        lsb->psi[i] = 1.0;
      } else {
        PetscScalar yitsi;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &yitsi));
        numer       = (1.0 - lsb->phi_scalar) * PetscRealPart(PetscConj(yitsi) * yitsi);
        lsb->psi[i] = numer / (numer + (lsb->phi_scalar * lsb->ytq[i] * lsb->stp[i]));
      }
    }
    lsb->needP = PETSC_FALSE;
  }

  /* Start the outer iterations for (B * X) */
  PetscCall(MatSymBrdnApplyJ0Fwd(B, X, Z));
  for (PetscInt i = 0; i < next - oldest; ++i) {
    Vec         s_i, y_i;
    PetscScalar yitsi;

    PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &yitsi));
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
    /* Compute the necessary dot products */
    PetscCall(VecDotBegin(s_i, Z, &stz));
    PetscCall(VecDotBegin(y_i, X, &ytx));
    PetscCall(VecDotEnd(s_i, Z, &stz));
    PetscCall(VecDotEnd(y_i, X, &ytx));
    /* Compute the pure BFGS component */
    PetscCall(VecAXPBYPCZ(Z, -PetscRealPart(stz) / lsb->stp[i], PetscRealPart(ytx) / yitsi, 1.0, lsb->P[i], y_i));
    /* Tack on the convexly scaled extras */
    PetscCall(VecAXPBYPCZ(lsb->work, 1.0 / yitsi, -1.0 / lsb->stp[i], 0.0, y_i, lsb->P[i]));
    PetscCall(VecDot(lsb->work, X, &wtx));
    PetscCall(VecAXPY(Z, lsb->psi[i] * lsb->stp[i] * PetscRealPart(wtx), lsb->work));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMSymBadBrdn(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM                  *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn               *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  MatLMVMSymBroydenScaleType stype;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVMSymBrdn(B, PetscOptionsObject));
  PetscCall(SymBroydenScalerGetType(lsb->rescale, &stype));
  if (stype == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(SymBroydenScalerSetDiagonalMode(lsb->rescale, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSymBadBrdn(Mat B)
{
  Mat_LMVM *lmvm;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBADBROYDEN));
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBadBrdn;
  B->ops->solve          = MatSolve_LMVMSymBadBrdn;

  lmvm            = (Mat_LMVM *)B->data;
  lmvm->ops->mult = MatMult_LMVMSymBadBrdn;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMSymBadBroyden - Creates a limited-memory Symmetric "Bad" Broyden-type matrix used
  for approximating Jacobians. L-SymBadBrdn is a convex combination of L-DFP and
  L-BFGS such that SymBadBrdn^{-1} = (1 - phi)*BFGS^{-1} + phi*DFP^{-1}. The combination factor
  phi is restricted to the range [0, 1], where the L-SymBadBrdn matrix is guaranteed
  to be symmetric positive-definite. Note that this combination is on the inverses and not
  on the forwards. For forward convex combinations, use the L-SymBrdn matrix.

  To use the L-SymBrdn matrix with other vector types, the matrix must be
  created using MatCreate() and MatSetType(), followed by `MatLMVMAllocate()`.
  This ensures that the internal storage and work vectors are duplicated from the
  correct type of vector.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Options Database Keys:
+ -mat_lmvm_phi        - (developer) convex ratio between BFGS and DFP components of the update
. -mat_lmvm_scale_type - (developer) type of scaling applied to J0 (none, scalar, diagonal)
. -mat_lmvm_theta      - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho        - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha      - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta       - (developer) exponential factor for the diagonal J0 scaling
- -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling

  Level: intermediate

  Note:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MatCreate()`, `MATLMVM`, `MATLMVMSYMBROYDEN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`
@*/
PetscErrorCode MatCreateLMVMSymBadBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSYMBADBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
