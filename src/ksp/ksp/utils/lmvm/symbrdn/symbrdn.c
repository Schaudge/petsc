#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/

/*------------------------------------------------------------*/

static PetscErrorCode UpdateP(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (lsb->needP) {
    /* Start the loop for (P[k] = (B_k) * S[k]) */
    PetscReal phi = lsb->phi;
    PetscInt  oldest, next;

    PetscCall(MatLMVMGetRange(B, &oldest, &next));
    for (PetscInt i = 0; i < next - oldest; ++i) {
      Vec s_i;

      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i));
      PetscCall(MatSymBrdnApplyJ0Fwd(B, s_i, lsb->P[i]));
      /* Compute the necessary dot products */
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar pjtsi, yjtsi;
        PetscReal   sjtpj = lsb->stp[j];
        PetscReal   yjtsj = lsb->rescale->yts[j];
        Vec         y_j;

        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_Y, &y_j));
        PetscCall(VecDot(s_i, lsb->P[j], &pjtsi));
        PetscCall(VecDot(s_i, y_j, &yjtsi));

        PetscScalar alpha = ((phi - 1.0) / sjtpj) * pjtsi - (phi / yjtsj) * yjtsi;
        PetscScalar beta  = -(phi / yjtsj) * pjtsi + ((yjtsj + phi * sjtpj) / (yjtsj * yjtsj)) * yjtsi;
        PetscCall(VecAXPBYPCZ(lsb->P[i], alpha, beta, 1.0, lsb->P[j], y_j));
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_Y, &y_j));
      }
      PetscCall(VecDotRealPart(s_i, lsb->P[i], &lsb->stp[i]));
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i));
    }
    lsb->needP = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The solution method below is the matrix-free implementation of
  Equation 8.6a in Dennis and More "Quasi-Newton Methods, Motivation
  and Theory" (https://epubs.siam.org/doi/abs/10.1137/1019005).

  Q[i] = (B_i)^{-1}*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatSolve without incurring redundant computation.

  dX <- J0^{-1} * F

  for i=0,1,2,...,k
    # Q[i] = (B_i)^T{-1} Y[i]

    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (S[i]^T F)
    zeta = 1.0 / (Y[i]^T Q[i])
    gamma = zeta * (Y[i]^T dX)

    dX <- dX - (gamma * Q[i]) + (alpha * Y[i])
    W <- (rho * S[i]) - (zeta * Q[i])
    dX <- dX + (psi[i] * (Y[i]^T Q[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatSolve_LMVMSymBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    PetscCall(MatSolve_LMVMBFGS(B, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (lsb->phi == 1.0) {
    PetscCall(MatSolve_LMVMDFP(B, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  PetscCall(UpdateP(B));
  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (PetscInt i = 0; i < next - oldest; ++i) {
      Vec y_i;

      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
      PetscCall(MatSymBrdnApplyJ0Inv(B, y_i, lsb->Q[i]));
      /* Compute the necessary dot products */
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar qjtyi, sjtyi;
        PetscReal   yjtqj = lsb->ytq[j];
        PetscReal   yjtsj = lsb->rescale->yts[j];
        PetscReal   psi   = lsb->psi[j];
        Vec         s_j;

        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_S, &s_j));
        PetscCall(VecDot(y_i, lsb->Q[j], &qjtyi));
        PetscCall(VecDot(y_i, s_j, &sjtyi));

        PetscScalar alpha = ((psi - 1.0) / yjtqj) * qjtyi - (psi / yjtsj) * sjtyi;
        PetscScalar beta  = -(psi / yjtsj) * qjtyi + ((yjtsj + psi * yjtqj) / (yjtsj * yjtsj)) * sjtyi;

        PetscCall(VecAXPBYPCZ(lsb->Q[i], alpha, beta, 1.0, lsb->Q[j], s_j));
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_S, &s_j));
      }
      PetscCall(VecDotRealPart(y_i, lsb->Q[i], &lsb->ytq[i]));
      if (lsb->phi == 1.0) {
        lsb->psi[i] = 0.0;
      } else if (lsb->phi == 0.0) {
        lsb->psi[i] = 1.0;
      } else {
        PetscReal numer;

        numer       = (1.0 - lsb->phi) * lsb->rescale->yts[i] * lsb->rescale->yts[i];
        lsb->psi[i] = numer / (numer + (lsb->phi * lsb->ytq[i] * lsb->stp[i]));
      }
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
    }
    lsb->needQ = PETSC_FALSE;
  }

  /* Start the outer iterations for ((B^{-1}) * dX) */
  PetscCall(MatSymBrdnApplyJ0Inv(B, F, dX));
  /* Get all the dot products we need */
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscReal   yitqi = lsb->ytq[i];
    PetscReal   yitsi = lsb->rescale->yts[i];
    PetscReal   psi   = lsb->psi[i];
    PetscScalar qitf, sitf;
    Vec         s_i;

    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i));
    PetscCall(VecDot(F, lsb->Q[i], &qitf));
    PetscCall(VecDot(F, s_i, &sitf));

    PetscScalar alpha = ((psi - 1.0) / yitqi) * qitf - (psi / yitsi) * sitf;
    PetscScalar beta  = -(psi / yitsi) * qitf + ((yitsi + psi * yitqi) / (yitsi * yitsi)) * sitf;
    PetscCall(VecAXPBYPCZ(dX, alpha, beta, 1.0, lsb->Q[i], s_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*
  The forward-product below is the matrix-free implementation of
  Equation 16 in Dennis and Wolkowicz "Sizing and Least Change Secant
  Methods" (http://www.caam.rice.edu/caam/trs/90/TR90-05.pdf).

  P[i] = (B_i)*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing P[i] terms in expensive nested-loops.

  Z <- J0 * X

  for i=0,1,2,...,k
    # P[i] = (B_k) * S[i]

    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (Y[i]^T F)
    zeta = 1.0 / (S[i]^T P[i])
    gamma = zeta * (S[i]^T dX)

    dX <- dX - (gamma * P[i]) + (alpha * S[i])
    W <- (rho * Y[i]) - (zeta * P[i])
    dX <- dX + (phi * (S[i]^T P[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatMult_LMVMSymBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscReal    phi  = lsb->phi;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    PetscCall(MatMult_LMVMBFGS(B, X, Z));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (lsb->phi == 1.0) {
    PetscCall(MatMult_LMVMDFP(B, X, Z));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(UpdateP(B));

  /* Start the outer iterations for (B * X) */
  PetscCall(MatSymBrdnApplyJ0Fwd(B, X, Z));

  /* Get all the dot products we need */
  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscReal   sitpi = lsb->stp[i];
    PetscReal   yitsi = lsb->rescale->yts[i];
    PetscScalar pitx, yitx;
    Vec         y_i;

    PetscCall(VecDot(X, lsb->P[i], &pitx));
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
    PetscCall(VecDot(X, y_i, &yitx));

    PetscScalar alpha = ((phi - 1.0) / sitpi) * pitx - (phi / yitsi) * yitx;
    PetscScalar beta  = -(phi / yitsi) * pitx + ((yitsi + phi * sitpi) / (yitsi * yitsi)) * yitx;
    PetscCall(VecAXPBYPCZ(Z, alpha, beta, 1.0, lsb->P[i], y_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt     old_k;
  PetscReal    curvtol, ststmp;
  PetscScalar  curvature, ytytmp;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));

    /* Test if the updates can be accepted */
    PetscCall(VecDotNorm2(lmvm->Fprev, lmvm->Xprev, &curvature, &ststmp));
    if (ststmp < lmvm->eps) curvtol = 0.0;
    else curvtol = lmvm->eps * ststmp;

    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good, accept it */
      lsb->watchdog = 0;
      lsb->needP = lsb->needQ = PETSC_TRUE;
      old_k                   = lmvm->k;
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      /* If we hit the memory limit, shift the yts, yty and sts arrays */
      if (old_k == lmvm->k) {
        for (PetscInt i = 0; i < next - oldest - 1; ++i) {
          lsb->rescale->yts[i] = lsb->rescale->yts[i + 1];
          lsb->rescale->yty[i] = lsb->rescale->yty[i + 1];
          lsb->rescale->sts[i] = lsb->rescale->sts[i + 1];
        }
      }
      /* Update history of useful scalars */
      lsb->rescale->yts[lmvm->k] = PetscRealPart(curvature);
      {
        Vec y_last;
        PetscCall(MatLMVMGetVecsRead(B, next, LMBASIS_Y, &y_last));
        PetscCall(VecDot(y_last, y_last, &ytytmp));
        PetscCall(MatLMVMRestoreVecsRead(B, next, LMBASIS_Y, &y_last));
        lsb->rescale->yty[lmvm->k] = PetscRealPart(ytytmp);
      }
      {
        lsb->rescale->sts[lmvm->k] = ststmp;
      }
      /* Compute the scalar scale if necessary */
      PetscCall(SymBroydenScalerUpdate(B, lsb->rescale));
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lsb->watchdog;
    }
  } else {
    PetscCall(SymBroydenScalerInitializeJ0(B, lsb->rescale));
  }

  if (lsb->watchdog > lsb->max_seq_rejects) { PetscCall(MatLMVMReset(B, PETSC_FALSE)); }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMSymBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM    *bdata = (Mat_LMVM *)B->data;
  Mat_SymBrdn *blsb  = (Mat_SymBrdn *)bdata->ctx;
  Mat_LMVM    *mdata = (Mat_LMVM *)M->data;
  Mat_SymBrdn *mlsb  = (Mat_SymBrdn *)mdata->ctx;

  PetscFunctionBegin;
  mlsb->phi     = blsb->phi;
  mlsb->needP   = blsb->needP;
  mlsb->needQ   = blsb->needQ;
  mlsb->useP    = blsb->useP;
  mlsb->useQ    = blsb->useQ;
  mlsb->use_stp = blsb->use_stp;
  mlsb->use_ytq = blsb->use_ytq;
  if (blsb->use_stp) PetscCall(PetscArraycpy(mlsb->stp, blsb->stp, bdata->k + 1));
  if (blsb->use_ytq) PetscCall(PetscArraycpy(mlsb->ytq, blsb->ytq, bdata->k + 1));
  for (PetscInt i = 0; i <= bdata->k; ++i) {
    if (blsb->useP) PetscCall(VecCopy(blsb->P[i], mlsb->P[i]));
    if (blsb->useQ) PetscCall(VecCopy(blsb->Q[i], mlsb->Q[i]));
  }
  mlsb->watchdog        = blsb->watchdog;
  mlsb->max_seq_rejects = blsb->max_seq_rejects;
  PetscCall(SymBroydenScalerCopy(blsb->rescale, mlsb->rescale, bdata->k + 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMSymBrdn_Internal(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&lsb->work));
  PetscCall(PetscFree3(lsb->stp, lsb->ytq, lsb->workscalar));
  PetscCall(PetscFree(lsb->psi));
  if (lsb->P) PetscCall(VecDestroyVecs(lmvm->m, &lsb->P));
  if (lsb->Q) PetscCall(VecDestroyVecs(lmvm->m, &lsb->Q));
  lsb->allocated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMSymBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  lsb->watchdog = 0;
  lsb->needP = lsb->needQ = PETSC_TRUE;
  PetscCall(SymBroydenScalerReset(B, lsb->rescale, destructive));
  if (lsb->allocated) {
    if (destructive) {
      PetscCall(MatReset_LMVMSymBrdn_Internal(B));
    } else {
      PetscCall(PetscMemzero(lsb->psi, lmvm->m));
      PetscCall(SymBroydenScalerInitializeJ0(B, lsb->rescale));
    }
  }
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMSymBrdn_Internal(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (!lsb->allocated) {
    PetscCall(VecDuplicate(lmvm->Xprev, &lsb->work));
    PetscCall(PetscMalloc3(lmvm->m, &lsb->stp, lmvm->m, &lsb->ytq, lmvm->m, &lsb->workscalar));
    PetscCall(PetscCalloc1(lmvm->m, &lsb->psi));
    if (lmvm->m > 0) {
      if (lsb->useP) PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->P));
      if (lsb->useQ) PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->Q));
    }
    PetscCall(SymBroydenScalerAllocate(B, lsb->rescale));
    lsb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAllocate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  PetscFunctionBegin;
  PetscCall(MatAllocate_LMVM(B, X, F));
  PetscCall(MatAllocate_LMVMSymBrdn_Internal(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMSymBrdn(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenGetPhi_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetPhi_C", NULL));
  PetscCall(SymBroydenScalerDestroy(&(lsb->rescale)));
  if (lsb->allocated) { PetscCall(MatReset_LMVMSymBrdn_Internal(B)); }
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMSymBrdn(Mat B)
{
  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  PetscCall(MatAllocate_LMVMSymBrdn_Internal(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVMSymBrdn(Mat B, PetscViewer pv)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscBool    isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) { PetscCall(PetscViewerASCIIPrintf(pv, "Convex factors: phi=%g\n", (double)lsb->phi)); }
  PetscCall(SymBroydenScalerView(lsb->rescale, pv));
  PetscCall(MatView_LMVM(B, pv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetFromOptions_LMVMSymBrdn(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "Restricted/Symmetric Broyden method for approximating SPD Jacobian actions (MATLMVMSYMBRDN)");
  PetscCall(PetscOptionsReal("-mat_lmvm_phi", "(developer) convex ratio between BFGS and DFP components of the update", "", lsb->phi, &lsb->phi, NULL));
  PetscCheck(!(lsb->phi < 0.0) && !(lsb->phi > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  PetscCall(SymBroydenScalerSetFromOptions(B, lsb->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatLMVMSymBroydenGetPhi - Get the phi parameter for a Broyden class quasi-Newton update matrix

  Input Parameter:
. B - The matrix

  Output Parameter:
. phi - a number defining an update that is a convex combination of the BFGS update (phi = 0) and DFP update (phi = 1)

  Level: developer

.seealso: [](chapter_ksp), `MATLMVMSYMBROYDEN`, `MATLMVMDFP`, `MATLMVMBFGS`, `MatLMVMSymBroydenSetPhi()`
@*/
PetscErrorCode MatLMVMSymBroydenGetPhi(Mat B, PetscReal *phi)
{
  PetscFunctionBegin;
  PetscUseMethod(B, "MatLMVMSymBroydenGetPhi_C", (Mat, PetscReal *), (B, phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBroydenGetPhi_SymBrdn(Mat B, PetscReal *phi)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  *phi = lsb->phi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSymBroydenSetPhi - Get the phi parameter for a Broyden class quasi-Newton update matrix

  Input Parameter:
+ B - The matrix
- phi - a number defining an update that is a convex combination of the BFGS update (phi = 0) and DFP update (phi = 1)

  Level: developer

.seealso: [](chapter_ksp), `MATLMVMSYMBROYDEN`, `MATLMVMDFP`, `MATLMVMBFGS`, `MatLMVMSymBroydenGetPhi()`
@*/
PetscErrorCode MatLMVMSymBroydenSetPhi(Mat B, PetscReal phi)
{
  PetscFunctionBegin;
  PetscTryMethod(B, "MatLMVMSymBroydenSetPhi_C", (Mat, PetscReal), (B, phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBroydenSetPhi_SymBrdn(Mat B, PetscReal phi)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  lsb->phi = phi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMSymBrdn(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *lsb;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBROYDEN));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE)); // TODO: change to HPD when available
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMSymBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBrdn;
  B->ops->setup          = MatSetUp_LMVMSymBrdn;
  B->ops->destroy        = MatDestroy_LMVMSymBrdn;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->square        = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMSymBrdn;
  lmvm->ops->reset    = MatReset_LMVMSymBrdn;
  lmvm->ops->update   = MatUpdate_LMVMSymBrdn;
  lmvm->ops->mult     = MatMult_LMVMSymBrdn;
  lmvm->ops->multht   = MatMult_LMVMSymBrdn;
  lmvm->ops->solve    = MatSolve_LMVMSymBrdn;
  lmvm->ops->solveht  = MatSolve_LMVMSymBrdn;
  lmvm->ops->copy     = MatCopy_LMVMSymBrdn;

  PetscCall(PetscNew(&lsb));
  lmvm->ctx      = (void *)lsb;
  lsb->allocated = PETSC_FALSE;
  lsb->needP = lsb->needQ = PETSC_TRUE;
  lsb->phi                = 0.125;
  lsb->watchdog           = 0;
  lsb->max_seq_rejects    = lmvm->m / 2;

  lsb->useP    = PETSC_TRUE;
  lsb->useQ    = PETSC_TRUE;
  lsb->use_stp = PETSC_TRUE;
  lsb->use_ytq = PETSC_TRUE;

  Vec J0inv;
  PetscCall(SymBroydenScalerCreate(&lsb->rescale));
  PetscCall(MatLMVMGetJ0InvDiag(B, &J0inv));
  PetscCall(VecSet(J0inv, 1.0));
  PetscCall(MatLMVMRestoreJ0InvDiag(B, &J0inv));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenGetPhi_C", MatLMVMSymBroydenGetPhi_SymBrdn));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetPhi_C", MatLMVMSymBroydenSetPhi_SymBrdn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatLMVMSymBroydenSetDelta - Sets the starting value for the diagonal scaling vector computed
  in the SymBrdn approximations (also works for BFGS and DFP).

  Input Parameters:
+ B     - LMVM matrix
- delta - initial value for diagonal scaling

  Level: intermediate

@*/
PetscErrorCode MatLMVMSymBroydenSetDelta(Mat B, PetscScalar delta)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscBool    is_bfgs, is_dfp, is_symbrdn, is_symbadbrdn;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMBFGS, &is_bfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDFP, &is_dfp));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBROYDEN, &is_symbrdn));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBADBROYDEN, &is_symbadbrdn));
  PetscCheck(is_bfgs || is_dfp || is_symbrdn || is_symbadbrdn, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "diagonal scaling is only available for DFP, BFGS and SymBrdn matrices");
  PetscCall(SymBroydenScalerSetDelta(lsb->rescale, PetscAbsReal(PetscRealPart(delta))));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatLMVMSymBroydenSetScaleType - Sets the scale type for symmetric Broyden-type updates.

  Input Parameters:
+   snes - the iterative context
-   rtype - restart type

  Options Database Key:
. -mat_lmvm_scale_type <none,scalar,diagonal> - set the scaling type

  Level: intermediate

  MatLMVMSymBrdnScaleTypes:
+   MAT_LMVM_SYMBROYDEN_SCALE_NONE - initial Hessian is the identity matrix
.   MAT_LMVM_SYMBROYDEN_SCALE_SCALAR - use the Shanno scalar as the initial Hessian
-   MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL - use a diagonalized BFGS update as the initial Hessian

.seealso: [](ch_ksp), `MATLMVMSYMBROYDEN`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatLMVMSymBroydenSetScaleType(Mat B, MatLMVMSymBroydenScaleType stype)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(SymBroydenScalerSetType(B, lsb->rescale, stype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMSymBroyden - Creates a limited-memory Symmetric Broyden-type matrix used
  for approximating Jacobians. L-SymBrdn is a convex combination of L-DFP and
  L-BFGS such that SymBrdn = (1 - phi)*BFGS + phi*DFP. The combination factor
  phi is restricted to the range [0, 1], where the L-SymBrdn matrix is guaranteed
  to be symmetric positive-definite.

  To use the L-SymBrdn matrix with other vector types, the matrix must be
  created using MatCreate() and MatSetType(), followed by MatLMVMAllocate().
  This ensures that the internal storage and work vectors are duplicated from the
  correct type of vector.

  Collective

  Input Parameters:
+ comm - MPI communicator, set to PETSC_COMM_SELF
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

  Options Database Keys:
+ -mat_lmvm_phi        - (developer) convex ratio between BFGS and DFP components of the update
. -mat_lmvm_scale_type - (developer) type of scaling applied to J0 (none, scalar, diagonal)
. -mat_lmvm_theta      - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho        - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha      - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta       - (developer) exponential factor for the diagonal J0 scaling
- -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling

  Level: intermediate

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMSYMBROYDEN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`
@*/
PetscErrorCode MatCreateLMVMSymBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSYMBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSymBrdnApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSymBrdnApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}
