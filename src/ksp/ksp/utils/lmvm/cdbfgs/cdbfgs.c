#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscis.h>
#include <petscoptions.h>


const char *const MatLBFGSTypes[] = {"basic", "cd_reorder", "cd_inplace", "MatLBFGSType", "MAT_LBFGS_", NULL};

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
  case MAT_LBFGS_CD_REORDER:
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
  if (lbfgs->Wfull != NULL) {
    PetscCall(MatMultTranspose(lbfgs->Wfull, F, lbfgs->work_0));
  } else {
    PetscCall(MatMultTranspose(lbfgs->Yfull, dX, lbfgs->work_0));
  }

  // work_1 = S^T * F
  PetscCall(MatMultTranspose(lbfgs->Sfull, F, lbfgs->work_1));

  // work_2 = Rbar^{-1} work_1
  PetscCall(MatSolveUpperTriangularRecycleOrder(lbfgs, lbfgs->StY, lbfgs->idx_begin, lbfgs->work_1, lbfgs->work_2));

  // work_1 = C * work_2 + work_0
  PetscCall(MatMultAdd(lbfgs->C, lbfgs->work_2, lbfgs->work_0, lbfgs->work_1));

  // work_0 = Rbar^{-T} work_1
  PetscCall(MatSolveUpperTriangularRecycleOrderTranspose(lbfgs, lbfgs->StY, lbfgs->idx_begin, lbfgs->work_1, lbfgs->work_0));

  // dX += S * work_0
  PetscCall(MatMultAdd(lbfgs->Sfull, lbfgs->work_0, dX, dX));

  // dX += S * work_0
  if (lbfgs->Wfull != NULL) {
    PetscCall(MatMultAdd(lbfgs->Wfull, lbfgs->work_2, dX, dX));
  } else {
    PetscCall(MatMult(lbfgs->Yfull, lbfgs->work_2, lbfgs->work_0));
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
  
  const PetscScalar *xx, *ff, *array_read;
  PetscScalar       curvature, ststmp, *buffer, *vals, *array_write;
  PetscReal         curvtol;
  PetscInt          n, low, high, *is_indices, i, j, N, sty_LDA, *rows;
  PetscMemType      memtype_sy;
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
      PetscCall(VecGetLocalSize(lmvm->Xprev, &n));
      PetscCall(VecGetOwnershipRange(lmvm->Xprev, &low, &high));
      PetscCall(PetscMalloc2(n, &lbfgs->idx_rows, 1, &lbfgs->idx_cols));
      for (i=low; i<high; i++) {
        lbfgs->idx_rows[i] = i;
      }

      switch (lbfgs->strategy) {
      case (MAT_LBFGS_CD_REORDER):
        if (lmvm->k == lmvm->m-1) {
          /* S Matrix is full. Shift matrix via memcpy */
          PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->Sfull, &array_read, &memtype_sy));
          // Assert(lda == m)
          PetscCall(PetscMalloc1(lmvm->m*lmvm->m - lmvm->m - 1, &buffer));
          PetscCall(PetscMemcpy(buffer, &array_read[lmvm->m+1], (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->Sfull, &array_read));
  
          PetscCall(MatDenseGetArrayWriteAndMemType(lbfgs->Sfull, &array_write, &memtype_sy));
          PetscCall(PetscMemcpy(array_write, &buffer, (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayWriteAndMemType(lbfgs->Sfull, &array_write));
          PetscCall(PetscFree(buffer));

          /* Y Matrix is full. Shift matrix via memcpy */
          PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->Yfull, &array_read, &memtype_sy));
          // Assert(lda == m)
          PetscCall(PetscMalloc1(lmvm->m*lmvm->m - lmvm->m - 1, &buffer));
          PetscCall(PetscMemcpy(buffer, &array_read[lmvm->m+1], (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->Yfull, &array_read));
  
          PetscCall(MatDenseGetArrayWriteAndMemType(lbfgs->Yfull, &array_write, &memtype_sy));
          PetscCall(PetscMemcpy(array_write, &buffer, (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayWriteAndMemType(lbfgs->Yfull, &array_write));
          PetscCall(PetscFree(buffer));
        } else {
          lmvm->k = lmvm->k + 1;
        }
        lbfgs->idx_cols[0] = lmvm->k;
        break;
      case (MAT_LBFGS_CD_INPLACE):
        /* Inplace doesn't move memory, but rather only finds index of oldest memory */
        if (lmvm->k == lmvm->m-1) {
          lbfgs->idx_begin = (lbfgs->idx_begin + 1) % lmvm->m;
          lbfgs->idx_cols[0] = lbfgs->idx_begin;
        } else {
          lmvm->k = lmvm->k + 1;
          lbfgs->idx_cols[0] = lmvm->k;
        }
        /* Generate the required row/col idx arrays for data transfer */
        break;
      case (MAT_LBFGS_BASIC):
        break;
      }

      /* First update the S^T matrix */
      PetscCall(VecGetArrayRead(lmvm->Xprev, &xx));
      PetscCall(MatSetValues(lbfgs->Sfull, n, lbfgs->idx_rows, 1, lbfgs->idx_cols, xx, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(lbfgs->Sfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->Sfull, MAT_FINAL_ASSEMBLY));
      PetscCall(VecRestoreArrayRead(lmvm->Xprev, &xx));
      /* Now repeat update for the Y^T matrix */
      PetscCall(VecGetArrayRead(lmvm->Fprev, &ff));
      PetscCall(MatSetValues(lbfgs->Yfull, n, lbfgs->idx_rows, 1, lbfgs->idx_cols, ff, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(lbfgs->Yfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->Yfull, MAT_FINAL_ASSEMBLY));
      PetscCall(VecRestoreArrayRead(lmvm->Xprev, &ff));
      /* Clean up unnecessary arrays */
      PetscCall(PetscFree2(lbfgs->idx_rows, lbfgs->idx_cols));

//      switch (lbfgs->strategy) {
//      case (MAT_LBFGS_CD_REORDER):
//        break;
//      case (MAT_LBFGS_CD_INPLACE):
//        break;
//      case (MAT_LBFGS_BASIC):
//        break;
//      }	      
      /* Factor StY = L + D + R */
      PetscCall(MatDestroy(&lbfgs->StYfull));
      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      PetscCall(MatConvert(lbfgs->StYfull, lbfgs->dense_type, MAT_INPLACE_MATRIX, &lbfgs->StYfull));//TODO is this needed, or done internally already?
      //TODO is there better way to do this??
#if 0      
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
#endif

      switch (lbfgs->strategy) {
      case (MAT_LBFGS_CD_REORDER):
        break;
      case (MAT_LBFGS_CD_INPLACE):
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
        break;
      case (MAT_LBFGS_BASIC):
        break;
      }	      

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
    PetscCall(MatDestroy(&lbfgs->Yfull));
    PetscCall(MatDestroy(&lbfgs->Sfull));
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
    if (N != M) SETERRQ(comm, PETSC_ERR_ARG_SIZ, "Incorrect problem sizes! dim(X) not equal to dim(F)");
    PetscCall(MatSetSizes(B, m, n, M, N));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));
    PetscCall(VecDuplicate(X, &lmvm->Xprev));
    PetscCall(VecDuplicate(F, &lmvm->Fprev));
    if (lmvm->m > 0) {
      /* Create iteration storage matrices */
      PetscCall(VecCreateMatDense(X, n, lmvm->m, N, lmvm->m, NULL, &lbfgs->Sfull));
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Yfull));
      MatType ttt;
      MatGetType(lbfgs->Sfull, &ttt);
      /* Create intermediate (sequential and small) matrices */
      //TODO: NOTE: "MMTM: This routine is currently only implemented for pairs of MATSEQAIJ matrices, for the MATSEQDENSE class, and for pairs of MATMPIDENSE matrices."
      PetscCall(MatMatTransposeMult(lbfgs->Sfull, lbfgs->Yfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StY));
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
    PetscCall(MatDestroy(&lbfgs->Sfull));
    PetscCall(MatDestroy(&lbfgs->Yfull));
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

PetscErrorCode MatSetFromOptions_LMVMCDBFGS(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt           alg;
  PetscBool          flg;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, NULL));
  PetscOptionsBegin(PetscObjectComm((PetscObject)B), NULL, "Compact dense BFGS method (MATLMVMCDBFGS)", NULL);
  PetscOptionsEnum("-mat_lbfgs_type", "Implementation options for L-BFGS", "MatLBFGSType", MatLBFGSTypes, (PetscEnum)lbfgs->strategy, (PetscEnum *)&lbfgs->strategy, NULL);
  PetscOptionsEnd();

  if (flg) lbfgs->strategy = (MatLBFGSType) MatLBFGSTypes[alg];

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
