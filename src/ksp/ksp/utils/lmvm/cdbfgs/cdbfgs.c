#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscblaslapack.h>
#include "petscmemtypeblas.h"
#include <petscmat.h>
#include <petscsys.h>
#include <petscis.h>
#include <petscoptions.h>


const char *const MatLBFGSTypes[] = {"basic", "cd_reorder", "cd_inplace", "MatLBFGSType", "MATLBFGS_", NULL};

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS       *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  } else {
    PetscCall(MatMult(lbfgs->diag_bfgs, X, Z));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS       *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  } else {
    PetscCall(MatSolve(lbfgs->diag_bfgs, F, dX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSolveUpperTriangularRecycleOrder(Mat_CDBFGS *lbfgs, Mat R, PetscInt lowest_index, Vec b, Vec x)
{
  PetscFunctionBegin;
  MPI_Comm comm = PetscObjectComm((PetscObject)R);
  switch (lbfgs->strategy) {
  case MATLBFGS_CD_REORDER:
    {
      // TODO: reorder of b
      const PetscScalar *r_array;
      PetscScalar *x_array;
      PetscMemType memtype_r, memtype_x;
      PetscInt lda, m = 10; // TODO: fix function signature to take LMVM instead of CDBFGS to get m

      PetscCall(VecCopy(b, x));
      PetscCall(MatDenseGetArrayReadAndMemType(R, &r_array, &memtype_r));
      PetscCall(VecGetArrayWriteAndMemType(x, &x_array, &memtype_x));
      PetscCall(MatDenseGetLDA(R, &lda));
      PetscAssert(memtype_x == memtype_r, comm, PETSC_ERR_PLIB, "Incompatible device pointers");
      switch (memtype_x) {
      case PETSC_MEMTYPE_HOST:
        /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
        {
          //PetscAssert(PetscDefined(BLAS)...));
          PetscScalar Alpha = 1.0;
          PetscBLASInt m_blas;
          PetscBLASInt one = 1;
          PetscCall(PetscBLASIntCast(m, &m_blas));
          PetscBLASInt lda_blas;
          PetscCall(PetscBLASIntCast(lda, &lda_blas));
          PetscBLASInt ldb_blas = lda_blas;
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
        }
        break;
      default:
        SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
      }
      PetscCall(VecRestoreArrayWriteAndMemType(x, &x_array));
      PetscCall(MatDenseRestoreArrayReadAndMemType(R, &r_array));
    }
    // TODO: mimic this blas call with cuBLAS / hipBLAS / rocmBLAS
    break;
  //case MATLBFGS_CD_INPLACE:
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented L-BFGS strategy");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveUpperTriangularRecycleOrderTranspose(Mat_CDBFGS *lbfgs, Mat R, PetscInt lowest_index, Vec b, Vec x)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode VecPlaceArrayMemType(Vec, PetscScalar *);
PetscErrorCode VecResetArrayMemType(Vec);

#if 0
static PetscErrorCode MatSolve_LMVMCDBFGS_Basic(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  const PetscScalar *S;
  const PetscScalar *Y;
  PetscInt n_local, lda;
  PetscCall(MatGetLocalSize(lbfgs->S, &n_local, NULL));
  PetscCall(MatDenseGetLDA(lbfgs->S, &lda));
  Vec q = lbfgs->work_0;
  PetscCall(VecCopy(F, lbfgs->work_0));
  Vec s = lbfgs->work_1;
  Vec y = lbfgs->work_2;
  PetscScalar *alpha = lbfgs->alphas;
  PetscScalar *beta = lbfgs->betas;
  PetscScalar *StY = lbfgs->StYdiag;
  PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->S, &S, NULL));
  PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->Y, &Y, NULL));
  // entrance loop
  for (PetscInt i = lmvm->k; i >= PetscMax(0,lmvm->k + 1 - lmvm->m); i--) {
    PetscInt col = i % lmvm->m;
    PetscCall(VecPlaceArrayMemType(s, (PetscScalar *) &S[col * lda]));
    PetscCall(VecPlaceArrayMemType(y, (PetscScalar *) &Y[col * lda]));
    PetscCall(VecDot(q, s, &alpha[col]));
    alpha[col] /= StY[col];
    PetscCall(VecAXPY(q, -alpha[col], y));
    PetscCall(VecResetArrayMemType(s));
    PetscCall(VecResetArrayMemType(y));
  }
  PetscCall(MatCDBFGSApplyJ0Inv(B, q, dX));
  for (PetscInt i = PetscMax(0,lmvm->k + 1 - lmvm->m); i <= lmvm->k; i++) {
    PetscInt col = i % lmvm->m;
    PetscCall(VecPlaceArrayMemType(s, (PetscScalar *) &S[col * lda]));
    PetscCall(VecPlaceArrayMemType(y, (PetscScalar *) &Y[col * lda]));
    PetscCall(VecDot(q, y, &beta[col]));
    beta[col] /= StY[col];
    PetscCall(VecAXPY(q, beta[col]-alpha[col], y));
    PetscCall(VecResetArrayMemType(s));
    PetscCall(VecResetArrayMemType(y));
  }
  PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->S, &S));
  PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->Y, &Y));

  // exit loop
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMCDBFGS_Inplace(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  /* Start with the J0 term */
  PetscCall(MatCDBFGSApplyJ0Inv(B, F, dX));
  if (lmvm->k == -1) {
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  const PetscScalar *Q;
  const PetscScalar *S;
  PetscInt n_local, lda;
  PetscCall(MatGetLocalSize(lbfgs->S, &n_local, NULL));
  PetscCall(MatDenseGetLDA(lbfgs->S, &lda));
  PetscCall(VecCopy(F, lbfgs->work_0));
  PetscScalar *QtF;
  PetscScalar *StF;
  const PetscScalar *F_array;
  PetscMemType memtype;
  MPI_Comm     comm = PetscObjectComm((PetscObject)B);
  PetscMPIInt  rank;
  PetscCall(MPI_Comm_rank(comm, &rank));
  PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->S, &S, &memtype));
  PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->Q, &Q, NULL));
  PetscCall(VecGetArrayWriteAndMemType(lbfgs->work_0, &QtF, NULL));
  PetscCall(VecGetArrayWriteAndMemType(lbfgs->work_1, &StF, NULL));
  PetscInt limit = PetscMin(lmvm->m, lmvm->k + 1);
  PetscCall(VecGetArrayReadAndMemType(F, &F_array, NULL));
  PetscCall(PetscMemTypeGEMV(memtype, "ConjugateTranspose", n_local, limit, 1.0, Q, lda, F_array, 1, 0.0, QtF, 1));
  PetscCall(PetscMemTypeGEMV(memtype, "ConjugateTranspose", n_local, limit, 1.0, S, lda, F_array, 1, 0.0, StF, 1));
  PetscCall(VecRestoreArrayReadAndMemType(F, &F_array));
  PetscCallMPI(MPI_Reduce(rank == 0 ? MPI_IN_PLACE : QtF, QtF, limit, MPIU_SCALAR, MPI_SUM, 0, comm));
  PetscCallMPI(MPI_Reduce(rank == 0 ? MPI_IN_PLACE : StF, StF, limit, MPIU_SCALAR, MPI_SUM, 0, comm));
  if (rank == 0) {
    const PetscScalar *C;
    const PetscScalar *StY;
    PetscInt ldc, ldsty;
    PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->C, &C, NULL));
    PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->StY, &StY, NULL));
    PetscCall(MatDenseGetLDA(lbfgs->C, &ldc));
    PetscCall(MatDenseGetLDA(lbfgs->StY, &ldsty));

    PetscInt hi = lmvm->k % lmvm->m; // location of the latest vector
    PetscInt lo = PetscMax(0, lmvm->k + 1 - lmvm->m) % lmvm->m; // location of the earliest vector

    // StF <- - R^{-1} StF
    PetscCall(PetscMemTypeTRSM(memtype, "Left", "Upper", "NoTranspose", "NotUnit", hi, 1, -1.0, StY, ldsty, StF, lmvm->m));
    if (lo != 0) {
      PetscCall(PetscMemTypeGEMV(memtype, "NoTranspose", lmvm->m - lo, hi, 1.0, &StY[lo], ldsty, StF, 1, -1.0, &StF[lo], 1));
      PetscCall(PetscMemTypeTRSM(memtype, "Left", "Upper", "NoTranspose", "NotUnit", lmvm->m - lo, 1, -1.0, &StY[lo * (ldsty + 1)], ldsty, &StF[lo], lmvm->m));
    }

    // QtF <- QtF + C * StF
    PetscCall(PetscMemTypeGEMV(memtype, "NoTranspose", limit, limit, 1.0, C, ldc, StF, lmvm->m, 1.0, QtF, 1));

    // QtF <- - R^{-T} QtF
    if (lo != 0) {
      PetscCall(PetscMemTypeTRSM(memtype, "Left", "Upper", "ConjugateTranspose", "NotUnit", lmvm->m - lo, 1, -1.0, &StY[lo * (ldsty + 1)], ldsty, &QtF[lo], lmvm->m));
      PetscCall(PetscMemTypeGEMV(memtype, "ConjugateTranspose", lmvm->m - lo, hi, 1.0, &StY[lo], ldsty, &QtF[lo], 1, -1.0, QtF, 1));
    }
    PetscCall(PetscMemTypeTRSM(memtype, "Left", "Upper", "ConjugateTranspose", "NotUnit", hi, 1, -1.0, StY, ldsty, QtF, lmvm->m));


    PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->StY, &StY));
    PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->C, &C));
  }
  PetscCallMPI(MPI_Bcast(QtF, limit, MPIU_SCALAR, 0, comm));
  PetscCallMPI(MPI_Bcast(StF, limit, MPIU_SCALAR, 0, comm));
  PetscScalar *dX_array;
  PetscCall(VecGetArrayWriteAndMemType(dX, &dX_array, NULL));
  PetscCall(PetscMemTypeGEMV(memtype, "NoTranspose", n_local, limit, 1.0, Q, lda, StF, 1, 1.0, dX_array, 1));
  PetscCall(PetscMemTypeGEMV(memtype, "NoTranspose", n_local, limit, 1.0, S, lda, QtF, 1, 1.0, dX_array, 1));
  PetscCall(VecRestoreArrayWriteAndMemType(dX, &dX_array));

  PetscCall(VecRestoreArrayWriteAndMemType(lbfgs->work_0, &QtF));
  PetscCall(VecRestoreArrayWriteAndMemType(lbfgs->work_1, &StF));
  PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->S, &S));
  PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->Q, &Q));

  // exit loop
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  /* Start with the J0 term */
  PetscCall(MatCDBFGSApplyJ0Inv(B, F, dX));
  if (lmvm->k == -1) {
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  // work_0 = Y^T * J0 * F
  if (lbfgs->H0_Y != NULL) {
    
  } else {
    PetscCall(MatMultTranspose(lbfgs->Y, dX, lbfgs->work_0));
  }

  // work_1 = S^T * F
  PetscCall(MatMultTranspose(lbfgs->S, F, lbfgs->work_1));

  // work_2 = Rbar^{-1} work_1
  PetscCall(MatSolveUpperTriangularRecycleOrder(lbfgs, lbfgs->StY, lbfgs->idx_begin, lbfgs->work_1, lbfgs->work_2));

  // work_1 = C * work_2 + work_0
  PetscCall(MatMultAdd(lbfgs->C, lbfgs->work_2, lbfgs->work_0, lbfgs->work_1));

  // work_0 = Rbar^{-T} work_1
  PetscCall(MatSolveUpperTriangularRecycleOrderTranspose(lbfgs, lbfgs->StY, lbfgs->idx_begin, lbfgs->work_1, lbfgs->work_0));

  // dX += S * work_0
  PetscCall(MatMultAdd(lbfgs->S, lbfgs->work_0, dX, dX));

  // dX += S * work_0
  if (lbfgs->H0_Y != NULL) {
    PetscCall(MatMultAdd(lbfgs->H0_Y, lbfgs->work_2, dX, dX));
  } else {
    PetscCall(MatMult(lbfgs->Y, lbfgs->work_2, lbfgs->work_0));
    PetscCall(MatCDBFGSApplyJ0Inv(B, lbfgs->work_0, lbfgs->work_1));
    PetscCall(VecAXPY(dX, 1.0, lbfgs->work_1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMCDBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Start with the B0 term */
  PetscCall(MatCDBFGSApplyJ0Fwd(B, X, Z));
  if (lmvm->k == -1) {
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  /* Negate the Z vector so that we can do summations and negate again at the end */
  PetscCall(VecScale(Z, -1.0));

  /* Apply Phi^T = [S^TB; Y^t] to incoming vector X */
  /* The result is stored in two halves, (rwork1 = S^T B X) and (rwork2 = Y^T X) */
#if 0
  PetscCall(MatCDBFGSApplyJ0Fwd(B, X, lbfgs->lwork1));
  PetscCall(MatMult(lbfgs->ST, lbfgs->lwork1, lbfgs->rwork1));
  PetscCall(MatMult(lbfgs->YT, X, lbfgs->rwork2));

  /* Start with the upper half of M */
  PetscCall(MatMultTranspose(lbfgs->ST, lbfgs->rwork1, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Fwd(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(MatMult(lbfgs->ST, lbfgs->lwork2, lbfgs->rwork3));
  PetscCall(MatMultTransposeAdd(lbfgs->L, lbfgs->rwork2, lbfgs->rwork3, lbfgs->rwork3));
  PetscCall(MatMultTranspose(lbfgs->ST, lbfgs->rwork3, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Fwd(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(VecAXPY(Z, 1.0, lbfgs->lwork2));

  /* Now bottom half of M */
  PetscCall(MatMult(lbfgs->D, lbfgs->rwork2, lbfgs->rwork4));
  PetscCall(VecScale(lbfgs->rwork4, -1.0));
  PetscCall(MatMultAdd(lbfgs->L, lbfgs->rwork1, lbfgs->rwork4, lbfgs->rwork4));
  PetscCall(MatMultTransposeAdd(lbfgs->YT, lbfgs->rwork4, Z, Z));

  /* Negate the output vector again for final result */
  PetscCall(VecScale(Z, -1.0));
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *dctx;
  
  const PetscScalar *xx, *ff, *vals;
  PetscScalar       curvature, ststmp, *sty_array;
  PetscReal         curvtol;
  PetscInt          n, low, high, *is_indices, i, j, N, sty_LDA;
  MatFactorInfo     info;
  IS                shift_is, full_is;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Test if the updates can be accepted */
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp));
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good, accept it */
      lbfgs->watchdog = 0;
      if (lmvm->k == lmvm->m-1) {
	lbfgs->idx_begin = (lbfgs->idx_begin + 1) % lmvm->m;
      } else {
        lmvm->k = lmvm->k + 1;
      }

#if 0
      /* Generate the required row/col idx arrays for data transfer */
      PetscCall(VecGetLocalSize(lmvm->Xprev, &n));
      PetscCall(VecGetOwnershipRange(lmvm->Xprev, &low, &high));
      PetscCall(PetscMalloc2(1, &lbfgs->idx_rows, n, &lbfgs->idx_cols));
      lbfgs->idx_rows[0] = lbfgs->idx_begin;
      for (i=low; i<high; i++) {
        lbfgs->idx_cols[i] = i;
      }
      /* First update the S^T matrix */
      PetscCall(VecGetArrayRead(lmvm->Xprev, &xx));
      PetscCall(MatSetValues(lbfgs->STfull, 1, lbfgs->idx_rows, n, lbfgs->idx_cols, xx, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(lbfgs->STfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->STfull, MAT_FINAL_ASSEMBLY));
      PetscCall(VecRestoreArrayRead(lmvm->Xprev, &xx));
      /* Now repeat update for the Y^T matrix */
      PetscCall(VecGetArrayRead(lmvm->Fprev, &ff));
      PetscCall(MatSetValues(lbfgs->YTfull, 1, lbfgs->idx_rows, n, lbfgs->idx_cols, ff, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(lbfgs->YTfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->YTfull, MAT_FINAL_ASSEMBLY));
      PetscCall(VecRestoreArrayRead(lmvm->Xprev, &ff));
      /* Clean up unnecessary arrays */
      PetscCall(PetscFree2(lbfgs->idx_rows, lbfgs->idx_cols));

      // pseudocode for the StYfull updated for CD_REORDER
      // MatDenseGetArrayReadAndMemType(StYfull, &StY, ...)
      // Assert(lda == m)
      // PetscMemcpy(StYwork, &StY[m+1], m^2 - (m + 1)) // take into account the memtype
      // PetscMemcpy(StY, StYwork, m^2 - (m + 1)) // take into account the memtype
      //

      /* Factor StY = L + D + R */
      PetscCall(MatDestroy(&lbfgs->StYfull));
      PetscCall(MatMatTransposeMult(lbfgs->STfull, lbfgs->YTfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      PetscCall(MatConvert(lbfgs->StYfull, lbfgs->dense_type, MAT_INPLACE_MATRIX, &lbfgs->StYfull));//TODO is this needed, or done internally already?
      //TODO is there better way to do this??
      for (i=0; i<lmvm->m; i++) {
        PetscCall(MatGetRow(lbfgs->StYfull, i, &n, NULL, &vals));
        for (j=0; j<n; j++) {
          if (i <= j) {
            PetscCall(MatSetValue(lbfgs->Rfull, i, j, vals[j], INSERT_VALUES));
            if (i == j) {
              PetscCall(MatSetValue(lbfgs->Dfull, i, j, vals[j], INSERT_VALUES));
            }
          } else {
            PetscCall(MatSetValue(lbfgs->Lfull, i, j, vals[j], INSERT_VALUES));
          }
        }
        PetscCall(MatRestoreRow(lbfgs->StYfull, i, &n, NULL, &vals));
      }
      PetscCall(MatAssemblyBegin(lbfgs->Lfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->Lfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyBegin(lbfgs->Dfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->Dfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyBegin(lbfgs->Rfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->Rfull, MAT_FINAL_ASSEMBLY));
      /* Clear out the previously formed submatrices and work vectors */
      PetscCall(MatDestroy(&lbfgs->ST));
      PetscCall(MatDestroy(&lbfgs->YT));
      PetscCall(MatDestroy(&lbfgs->StY));
      PetscCall(MatDestroy(&lbfgs->L));
      PetscCall(MatDestroy(&lbfgs->D));
      PetscCall(MatDestroy(&lbfgs->R));
      PetscCall(MatDestroy(&lbfgs->Rinv));
      PetscCall(VecDestroy(&lbfgs->rwork1));
      PetscCall(VecDestroy(&lbfgs->rwork2));
      PetscCall(VecDestroy(&lbfgs->rwork3));
      PetscCall(VecDestroy(&lbfgs->rwork4));

      /* Setting IS for shift */
      PetscCall(VecGetSize(X, &N));
      PetscCall(PetscMalloc1(lmvm->k+1, &is_indices));
      for (i = 0; i < lmvm->k+1; i++) {
        is_indices[i] = (i + lbfgs->idx_begin) % (lmvm->k+1); 
      }
      PetscCall(ISCreate(PetscObjectComm((PetscObject)B), &shift_is));
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)B), N, 0, 1, &full_is));
      PetscCall(ISSetType(shift_is, ISGENERAL));
      PetscCall(ISGeneralSetIndices(shift_is, lmvm->k+1, is_indices, PETSC_OWN_POINTER));
      /* Generate submatrices that span only the stored iterates */
      /* Unshifted, would-be-original matrix indexing:           */
      /* | 1 , 2 | */
      /* | 3 , 4 | */
      /* Shifted, reality matrix indexing:  */
      /* | 4 , 3 | */
      /* | 2 , 1 | */
      //PLAN 1: 
      //PetscCall(MatDenseGetArrayAndMemType(lbfgs->StYfull, &sty_array, NULL));
      //PetscCall(MatDenseGetLDA(lbfgs->StYfull, &sty_LDA));
      //PetscCall(VecCreateMatDense(X, lmvm->m, n, lmvm->m, sty_LDA, sty_array + lbfgs->idx_begin*(sty_LDA+1) , &lbfgs->StYfull_1));//TODO not sure about global-local issue here?
      //PetscCall(VecCreateMatDense(X, lmvm->m, n, lmvm->m, sty_LDA, sty_array + lbfgs->idx_begin*sty_LDA, &lbfgs->StYfull_2));
      //PetscCall(VecCreateMatDense(X, lmvm->m, n, lmvm->m, sty_LDA, sty_array + lbfgs->idx_begin, &lbfgs->StYfull_3));
      //PetscCall(VecCreateMatDense(X, lmvm->m, n, lmvm->m, sty_LDA, sty_array, &lbfgs->StYfull_4));
      //PetscCall(PetscFree(is_indices));
      //PetscCall(ISDestroy(&shift_is));
      //PetscCall(MatDenseRestoreArrayAndMemType(lbfgs->StYfull, &sty_array));
      ////Block matrices pointers. Now get ST,YT,L,D,R
      //// On second thought, too cumbersome. Below option is prob better?

      //PLAN 2:
      //I imagine MatSubMatrixVirtualUpdate would be better, but... IS size is potentially non-static.
      PetscCall(MatCreateSubMatrixVirtual(lbfgs->STfull, shift_is, full_is, &lbfgs->ST));
      PetscCall(MatCreateSubMatrixVirtual(lbfgs->YTfull, shift_is, full_is, &lbfgs->YT));
      PetscCall(MatCreateSubMatrixVirtual(lbfgs->StYfull, shift_is, shift_is, &lbfgs->StY));
      PetscCall(MatCreateSubMatrixVirtual(lbfgs->Lfull, shift_is, shift_is, &lbfgs->L));
      PetscCall(MatCreateSubMatrixVirtual(lbfgs->Dfull, shift_is, shift_is, &lbfgs->D));
      PetscCall(MatCreateSubMatrixVirtual(lbfgs->Rfull, shift_is, shift_is, &lbfgs->R));
      PetscCall(MatCreateSubMatrix(lbfgs->Rfull, shift_is, shift_is, MAT_INITIAL_MATRIX, &lbfgs->Rinv));

      PetscCall(ISDestroy(&full_is));
      /* Generate the work vectors from the submatrices */
      PetscCall(MatCreateVecs(lbfgs->R, &lbfgs->rwork1, &lbfgs->rwork2));
      PetscCall(MatCreateVecs(lbfgs->R, &lbfgs->rwork3, &lbfgs->rwork4));
      /* Factor the R matrix for inversion */
      //PetscCall(MatGetOrdering(lbfgs->Rinv, MATORDERINGRCM, &perm, &iperm));
      PetscCall(MatFactorInfoInitialize(&info));
      info.fill = 0.0;
      info.dtcol = 0.0;
      info.zeropivot = 1e-12;
      info.pivotinblocks = 0.0;
      PetscCall(MatLUFactor(lbfgs->Rinv, NULL, NULL, &info));
      //TODO factorization for DenseMat?
      //PetscCall(MatLUFactor(lbfgs->Rinv, perm, iperm, &info));
      /* Update the diagonal H0 if it exists */
      if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
        PetscCall(MatLMVMUpdate(lbfgs->diag_bfgs, X, F));
      }
#endif
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lbfgs->watchdog;
      lmvm->k = lmvm->k - 1;
    }
  } else {
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      /* No previous updates have been set, so we just update the diagonal with an initial scalar */
      dbase = (Mat_LMVM*)lbfgs->diag_bfgs->data;
      dctx = (Mat_DiagBrdn*)dbase->ctx;
      PetscCall(VecSet(dctx->invD, lbfgs->delta));
    }
  }
  
  if (lbfgs->watchdog > lbfgs->max_seq_rejects) {
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      PetscCall(MatLMVMReset(lbfgs->diag_bfgs, PETSC_FALSE));
    }
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMCDBFGS(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *blbfgs = (Mat_CDBFGS*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_CDBFGS        *mlbfgs = (Mat_CDBFGS*)mdata->ctx;
  
  PetscFunctionBegin;
  mlbfgs->watchdog        = blbfgs->watchdog;
  mlbfgs->max_seq_rejects = blbfgs->max_seq_rejects;
  if (!(bdata->J0 || bdata->user_pc || bdata->user_ksp || bdata->user_scale)) {
    PetscCall(MatCopy(blbfgs->diag_bfgs, mlbfgs->diag_bfgs, SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMCDBFGS(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  lbfgs->watchdog = 0;
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    PetscCall(MatLMVMReset(lbfgs->diag_bfgs, destructive));
  }
  if (lbfgs->allocated && destructive) {
    PetscCall(MatDestroy(&lbfgs->StY));
    PetscCall(MatDestroy(&lbfgs->Y));
    PetscCall(MatDestroy(&lbfgs->S));
    PetscCall(MatDestroy(&lbfgs->H0_Y));
    PetscCall(VecDestroy(&lbfgs->work_0));
    PetscCall(VecDestroy(&lbfgs->work_1));
    PetscCall(VecDestroy(&lbfgs->work_2));
    lbfgs->allocated = PETSC_FALSE;
  }
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscBool         same, allocate = PETSC_FALSE;
  VecType           vec_type;
  PetscInt          m, n, M, N;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  
  PetscFunctionBegin;
  if (lmvm->allocated) {
    PetscCall(VecGetType(X, &vec_type));
    PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->Xprev, vec_type, &same));
    if (!same) {
      /* Given X vector has a different type than allocated X-type data structures.
         We need to destroy all of this and duplicate again out of the given vector. */
      allocate = PETSC_TRUE;
      PetscCall(MatLMVMReset(B, PETSC_TRUE));
    } else {
      VecCheckMatCompatible(B, X, 2, F, 3);
    }
  } else {
    allocate = PETSC_TRUE;
  }
  if (allocate) {
    PetscCall(VecGetLocalSize(X, &n));
    PetscCall(VecGetSize(X, &N));
    PetscCall(VecGetLocalSize(F, &m));
    PetscCall(VecGetSize(F, &M));
    PetscAssert(N == M, comm, PETSC_ERR_ARG_SIZ, "Incorrect problem sizes! dim(X) not equal to dim(F)");
    PetscCall(MatSetSizes(B, m, n, M, N));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));
    PetscCall(VecDuplicate(X, &lmvm->Xprev));
    PetscCall(VecDuplicate(F, &lmvm->Fprev));


    if (lmvm->m > 0) {
      PetscInt mM = lmvm->m;
      PetscMPIInt rank;
      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscInt mm = (rank == 0) ? M : 0;

      /* Create iteration storage matrices */
      PetscCall(VecCreateMatDense(X, n, mm, N, mM, NULL, &lbfgs->S));
      PetscCall(VecCreateMatDense(F, m, mm, M, mM, NULL, &lbfgs->Y));
      MatType ttt;
      MatGetType(lbfgs->S, &ttt);
      /* Create intermediate (sequential and small) matrices */
      //TODO: NOTE: "MMTM: This routine is currently only implemented for pairs of MATSEQAIJ matrices, for the MATSEQDENSE class, and for pairs of MATMPIDENSE matrices."
      PetscCall(MatMatTransposeMult(lbfgs->S, lbfgs->Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StY));
      //TODO check whether these matrices are actually dense
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->work_0));
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->work_1));
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->work_1));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      PetscCall(MatLMVMAllocate(lbfgs->diag_bfgs, X, F));
    }
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMCDBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  if (lbfgs->allocated) {
    PetscCall(MatDestroy(&lbfgs->StY));
    PetscCall(MatDestroy(&lbfgs->S));
    PetscCall(MatDestroy(&lbfgs->Y));
    PetscCall(VecDestroy(&lbfgs->work_0));
    PetscCall(VecDestroy(&lbfgs->work_1));
    PetscCall(VecDestroy(&lbfgs->work_2));
    lbfgs->allocated = PETSC_FALSE;
  }
  PetscCall(MatDestroy(&lbfgs->diag_bfgs));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMCDBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  
  PetscInt          m, n, M, N;
  PetscMPIInt       size;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  Vec               Xtmp, Ftmp;

  PetscFunctionBegin;
  PetscCall(MatGetSize(B, &M, &N));
  if (M == 0 && N == 0) SETERRQ(comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
    PetscCall(MPI_Comm_size(comm, &size));
    if (size == 1) {
      PetscCall(VecCreateSeq(comm, N, &Xtmp));
      PetscCall(VecCreateSeq(comm, M, &Ftmp));
    } else {
      PetscCall(MatGetLocalSize(B, &m, &n));
      PetscCall(VecCreateMPI(comm, n, N, &Xtmp));
      PetscCall(VecCreateMPI(comm, m, M, &Ftmp));
    }
    PetscCall(MatAllocate_LMVMCDBFGS(B, Xtmp, Ftmp));
    PetscCall(VecDestroy(&Xtmp));
    PetscCall(VecDestroy(&Ftmp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVMCDBFGS(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscBool         isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii));
  PetscCall(MatView_LMVM(B, pv));
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    PetscCall(MatView(lbfgs->diag_bfgs, pv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetFromOptions_LMVMCDBFGS(Mat B, PetscOptionItems *_items)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscBool          flg;
  PetscInt           strategy;
  
  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, NULL));
  PetscOptionsBegin(PetscObjectComm((PetscObject)B), NULL, "Compact dense BFGS method (MATLMVMCDBFGS)", NULL);
  PetscOptionsEList("-mat_lbfgs_type", "Implementation options for L-BFGS", "Mat", MatLBFGSTypes, 3, MatLBFGSTypes[lbfgs->strategy], &strategy, &flg);
  PetscOptionsEnd();
  if (flg) lbfgs->strategy = (MatLBFGSType)strategy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMCDBFGS(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_CDBFGS        *lbfgs;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMCDBFGS));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  B->ops->view = MatView_LMVMCDBFGS;
  B->ops->setup = MatSetUp_LMVMCDBFGS;
  B->ops->setfromoptions = MatSetFromOptions_LMVMCDBFGS;
  B->ops->destroy = MatDestroy_LMVMCDBFGS;
  
  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMCDBFGS;
  lmvm->ops->reset = MatReset_LMVMCDBFGS;
  lmvm->ops->update = MatUpdate_LMVMCDBFGS;
  lmvm->ops->mult = MatMult_LMVMCDBFGS;
  lmvm->ops->solve = MatSolve_LMVMCDBFGS;
  lmvm->ops->copy = MatCopy_LMVMCDBFGS;
  
  PetscCall(PetscNew(&lbfgs));
  lmvm->ctx = (void*)lbfgs;
  lbfgs->allocated       = PETSC_FALSE;
  lbfgs->idx_begin       = 0;
  lbfgs->watchdog        = 0;
  lbfgs->delta           = 1.0;
  lbfgs->delta_min       = 1e-7;
  lbfgs->delta_max       = 100.0;
  lbfgs->max_seq_rejects = lmvm->m/2;
  
  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &lbfgs->diag_bfgs));
  PetscCall(MatSetType(lbfgs->diag_bfgs, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOptionsPrefix(lbfgs->diag_bfgs, "J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMCDBFGS - Creates a compact dense representation of the limited-memory
   Broyden-Fletcher-Goldfarb-Shanno (BFGS) approximation to a Hessian. This compact 
   dense representation reduces the L-BFGS update to a series of matrix-vector products 
   with compact dense matrices in lieu of the conventional matrix-free two-loop 
   algorithm. For most problems on CPUs, this compact dense representation is not as
   fast as the matrix-free two-loop implementation provided via MATLMVMBFGS. However, 
   it may be faster on GPUs for large enough problems (note: requires CUDA).

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Level: advanced

.seealso: MatCreate(), MATLMVM, MATLMVMCDBFGS, MatCreateLMVMBFGS()
@*/
PetscErrorCode MatCreateLMVMCDBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMCDBFGS));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
