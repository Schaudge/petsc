#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*------------------------------------------------------------*/

static PetscErrorCode UpdateQ(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    PetscReal psi = lsb->psi_scalar;
    PetscInt  oldest, next;

    PetscCall(MatLMVMGetRange(B, &oldest, &next));
    for (PetscInt i = 0; i < next - oldest; ++i) {
      Vec y_i;

      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
      PetscCall(MatSymBrdnApplyJ0Inv(B, y_i, lsb->Q[i]));
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar qjtyi, sjtyi;
        PetscReal   yjtqj = lsb->ytq[j];
        PetscScalar sjtyj;
        Vec         s_j;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + j, &sjtyj));
        sjtyj = PetscConj(sjtyj);
        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_S, &s_j));
        PetscCall(VecDot(y_i, lsb->Q[j], &qjtyi));
        PetscCall(VecDot(y_i, s_j, &sjtyi));

        PetscScalar alpha = ((psi - 1.0) / yjtqj) * qjtyi - (psi / sjtyj) * sjtyi;
        PetscScalar beta  = -(psi / sjtyj) * qjtyi + ((sjtyj + psi * yjtqj) / (sjtyj * sjtyj)) * sjtyi;
        PetscCall(VecAXPBYPCZ(lsb->Q[i], alpha, beta, 1.0, lsb->Q[j], s_j));
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_S, &s_j));
      }
      PetscCall(VecDotRealPart(y_i, lsb->Q[i], &lsb->ytq[i]));
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
    }
    lsb->needQ = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSymBadBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscReal    psi  = lsb->psi_scalar;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (psi == 0.0) {
    PetscCall(MatSolve_LMVMDFP(B, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (psi == 1.0) {
    PetscCall(MatSolve_LMVMBFGS(B, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(UpdateQ(B));

  /* Start the outer iterations for ((B^{-1}) * dX) */
  PetscCall(MatSymBrdnApplyJ0Inv(B, F, dX));

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscReal   yitqi = lsb->ytq[i];
    PetscScalar sityi;
    PetscScalar qitf, sitf;
    Vec         s_i;

    PetscCall(VecDot(F, lsb->Q[i], &qitf));
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i));
    PetscCall(VecDot(F, s_i, &sitf));

    PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &sityi));
    sityi = PetscConj(sityi);
    PetscScalar alpha = ((psi - 1.0) / yitqi) * qitf - (psi / sityi) * sitf;
    PetscScalar beta  = -(psi / sityi) * qitf + ((sityi + psi * yitqi) / (sityi * sityi)) * sitf;
    PetscCall(VecAXPBYPCZ(dX, alpha, beta, 1.0, lsb->Q[i], s_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMSymBadBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscReal    psi  = lsb->psi_scalar;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (psi == 1.0) {
    PetscCall(MatMult_LMVMBFGS(B, X, Z));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (psi == 0.0) {
    PetscCall(MatMult_LMVMDFP(B, X, Z));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  PetscCall(UpdateQ(B));
  if (lsb->needP) {
    /* Start the loop for (P[k] = (B_k) * S[k]) */
    for (PetscInt i = 0; i < next - oldest; ++i) {
      Vec s_i;

      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i));
      PetscCall(MatSymBrdnApplyJ0Fwd(B, s_i, lsb->P[i]));
      /* Compute the necessary dot products */
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar pjtsi, yjtsi;
        PetscReal   sjtpj = lsb->stp[j];
        PetscScalar sjtyj;
        PetscReal   phi = lsb->phi[j];
        Vec         y_j;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + j, &sjtyj));
        sjtyj = PetscConj(sjtyj);
        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_Y, &y_j));
        PetscCall(VecDot(s_i, lsb->P[j], &pjtsi));
        PetscCall(VecDot(s_i, y_j, &yjtsi));

        PetscScalar alpha = ((phi - 1.0) / sjtpj) * pjtsi - (phi / sjtyj) * yjtsi;
        PetscScalar beta  = -(phi / sjtyj) * pjtsi + ((sjtyj + phi * sjtpj) / (sjtyj * sjtyj)) * yjtsi;

        PetscCall(VecAXPBYPCZ(lsb->P[i], alpha, beta, 1.0, lsb->P[j], y_j));
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_Y, &y_j));
      }
      PetscCall(VecDotRealPart(s_i, lsb->P[i], &lsb->stp[i]));
      if (psi == 1.0) {
        lsb->phi[i] = 0.0;
      } else if (psi == 0.0) {
        lsb->phi[i] = 1.0;
      } else {
        PetscScalar sityi;
        PetscReal   numer;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &sityi));
        sityi = PetscConj(sityi);
        numer       = (1.0 - psi) * PetscRealPart(PetscConj(sityi) * sityi);
        lsb->phi[i] = numer / (numer + (psi * lsb->stp[i] * lsb->ytq[i]));
      }
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i));
    }
    lsb->needP = PETSC_FALSE;
  }

  /* Start the outer iterations for (B * X) */
  PetscCall(MatSymBrdnApplyJ0Fwd(B, X, Z));
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscReal   sitpi = lsb->stp[i];
    PetscScalar sityi;
    PetscReal   phi = lsb->phi[i];
    PetscScalar pitx, yitx;
    Vec         y_i;

    PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &sityi));
    sityi = PetscConj(sityi);
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
    PetscCall(VecDot(X, lsb->P[i], &pitx));
    PetscCall(VecDot(X, y_i, &yitx));

    PetscScalar alpha = ((phi - 1.0) / sitpi) * pitx - (phi / sityi) * yitx;
    PetscScalar beta  = -(phi / sityi) * pitx + ((sityi + phi * sitpi) / (sityi * sityi)) * yitx;
    PetscCall(VecAXPBYPCZ(Z, alpha, beta, 1.0, lsb->P[i], y_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
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
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "Restricted/Symmetric Bad Broyden method for approximating SPD Jacobian actions (MATLMVMSYMBADBROYDEN)");
  PetscCall(PetscOptionsReal("-mat_lmvm_psi", "convex ratio between DFP and BFGS components of the update", "", lsb->psi_scalar, &lsb->psi_scalar, NULL));
  PetscCheck(lsb->psi_scalar >= 0.0 && lsb->psi_scalar <= 1.0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  PetscCall(SymBroydenScalerSetFromOptions(B, lsb->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscCall(SymBroydenScalerGetType(lsb->rescale, &stype));
  if (stype == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(SymBroydenScalerSetDiagonalMode(lsb->rescale, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSymBadBrdn(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *lsb;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBADBROYDEN));
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBadBrdn;
  B->ops->solve          = MatSolve_LMVMSymBadBrdn;

  lmvm            = (Mat_LMVM *)B->data;
  lmvm->ops->mult = MatMult_LMVMSymBadBrdn;

  lsb = (Mat_SymBrdn *)lmvm->ctx;
  lsb->psi_scalar = 0.124;
  lsb->phi_scalar = PETSC_DETERMINE;
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
