#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscmat.h>
#include <petscsys.h>

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

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  /* Start with the H0 term */
  PetscCall(MatCDBFGSApplyJ0Inv(B, F, dX));
  if (lmvm->k == -1) {
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  /* Apply the Phi^T = [Y^TH; S^T] to the RHS vector F */
  /* The result is stored in two halves, (rwork1 = Y^T H F) and (rwork2 = S^T F) */
  PetscCall(MatCDBFGSApplyJ0Inv(B, F, lbfgs->lwork1));
  PetscCall(MatMult(lbfgs->YT, lbfgs->lwork1, lbfgs->rwork1));
  PetscCall(MatMult(lbfgs->ST, F, lbfgs->rwork2));

  /* Calculate dX = HY R^{-T) rwork2 */
  /* This concludes operations with top half of M */
  PetscCall(MatSolve(lbfgs->Rinv, lbfgs->rwork2, lbfgs->rwork3));
  PetscCall(VecScale(lbfgs->rwork3, -1.0));
  PetscCall(MatMultTranspose(lbfgs->YT, lbfgs->rwork3, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Inv(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(VecAXPY(dX, 1.0, lbfgs->lwork2));

  /* Calculate rwork3 = -R^{-T} rwork1 */
  PetscCall(MatSolveTranspose(lbfgs->Rinv, lbfgs->rwork1, lbfgs->rwork3));
  PetscCall(VecScale(lbfgs->rwork3, -1.0));

  /* Calculate rwork3 += R^{-T}(D + YtHY)R^{-1} rwork2 */
  PetscCall(MatSolve(lbfgs->Rinv, lbfgs->rwork2, lbfgs->rwork4));
  PetscCall(MatMultTranspose(lbfgs->YT, lbfgs->rwork4, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Inv(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(MatMult(lbfgs->YT, lbfgs->lwork2, lbfgs->rwork2));
  PetscCall(MatMultAdd(lbfgs->D, lbfgs->rwork4, lbfgs->rwork2, lbfgs->rwork2));
  PetscCall(MatSolveTransposeAdd(lbfgs->Rinv, lbfgs->rwork2, lbfgs->rwork3, lbfgs->rwork3));
  
  /* Calculate dX += S rwork3 */
  /* This concludes operations with bottom half of M */
  PetscCall(MatMultTransposeAdd(lbfgs->ST, lbfgs->rwork3, dX, dX));
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
  PetscScalar       *buffer, curvature, ststmp;
  PetscReal         curvtol;
  const PetscInt    *cols;
  PetscInt          n, low, high, *rows, i, j;
  IS                active_rows;
  MatFactorInfo     info;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

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
        /* There is no space left for new iterate so we have to shift and make room */
        PetscCall(PetscMalloc1(1, &rows));
        PetscCall(PetscMalloc1(B->rmap->n, &buffer));
        for (i=0; i<lmvm->k; i++) {
          rows[0] = i+1;
          /* Take the information one row ahead of the current idx */
          PetscCall(MatGetRow(lbfgs->STfull, i+1, &n, &cols, &vals));
          /* Copy the info into a buffer array and restore the row */
          for (j=0; j<n; j++) buffer[j] = vals[j];
          PetscCall(MatRestoreRow(lbfgs->STfull, i+1, &n, &cols, &vals));
          /* Place the info from the next row into this one, overwriting existing info */
          /* This process ultimately discards the information stored in the first row at idx 0 */
          /* New information can then be written into idx=lmvm->k */
          PetscCall(MatSetValues(lbfgs->STfull, 1, rows, n, cols, buffer, INSERT_VALUES));
          /* Repeat for the Y matrix */
          PetscCall(MatGetRow(lbfgs->YTfull, i+1, &n, &cols, &vals));
          for (j=0; j<n; j++) buffer[j] = vals[j];
          PetscCall(MatRestoreRow(lbfgs->YTfull, i+1, &n, &cols, &vals));
          PetscCall(MatSetValues(lbfgs->YTfull, 1, rows, n, cols, buffer, INSERT_VALUES));
        }
        PetscCall(PetscFree(rows));
        PetscCall(PetscFree(buffer));
      } else {
        lmvm->k = lmvm->k + 1;
      }
      /* Generate the required row/col idx arrays for data transfer */
      PetscCall(VecGetLocalSize(lmvm->Xprev, &n));
      PetscCall(VecGetOwnershipRange(lmvm->Xprev, &low, &high));
      PetscCall(PetscMalloc2(1, &lbfgs->idx_rows, n, &lbfgs->idx_cols));
      lbfgs->idx_rows[0] = lmvm->k;
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
      /* Factor StY = L + D + R */
      PetscCall(MatDestroy(&lbfgs->StYfull));
      PetscCall(MatMatTransposeMult(lbfgs->STfull, lbfgs->YTfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      PetscCall(MatConvert(lbfgs->StYfull, lbfgs->dense_type, MAT_INPLACE_MATRIX, &lbfgs->StYfull));
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
      PetscCall(VecDestroy(&lbfgs->rwork1));
      PetscCall(VecDestroy(&lbfgs->rwork2));
      PetscCall(VecDestroy(&lbfgs->rwork3));
      PetscCall(VecDestroy(&lbfgs->rwork4));
      /* Generate submatrices that span only the stored iterates */
      PetscCall(ISCreateStride(comm, lmvm->k+1, 0, 1, &active_rows));
      PetscCall(MatCreateSubMatrix(lbfgs->STfull, active_rows, NULL, MAT_INITIAL_MATRIX, &lbfgs->ST));
      PetscCall(MatCreateSubMatrix(lbfgs->YTfull, active_rows, NULL, MAT_INITIAL_MATRIX, &lbfgs->YT));
      PetscCall(MatCreateSubMatrix(lbfgs->StYfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->StY));
      PetscCall(MatCreateSubMatrix(lbfgs->Lfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->L));
      PetscCall(MatCreateSubMatrix(lbfgs->Dfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->D));
      PetscCall(MatCreateSubMatrix(lbfgs->Rfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->R));
      PetscCall(ISDestroy(&active_rows));
      /* Generate the work vectors from the submatrices */
      PetscCall(MatCreateVecs(lbfgs->R, &lbfgs->rwork1, &lbfgs->rwork2));
      PetscCall(MatCreateVecs(lbfgs->R, &lbfgs->rwork3, &lbfgs->rwork4));
      /* Factor the R matrix for inversion */
      PetscCall(MatDestroy(&lbfgs->Rinv));
      PetscCall(MatDuplicate(lbfgs->R, MAT_COPY_VALUES, &lbfgs->Rinv));
      //PetscCall(MatGetOrdering(lbfgs->Rinv, MATORDERINGRCM, &perm, &iperm));
      PetscCall(MatFactorInfoInitialize(&info));
      info.fill = 0.0;
      info.dtcol = 0.0;
      info.zeropivot = 1e-12;
      info.pivotinblocks = 0.0;
      PetscCall(MatLUFactor(lbfgs->Rinv, NULL, NULL, &info));
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
    PetscCall(MatDestroy(&lbfgs->ST));
    PetscCall(MatDestroy(&lbfgs->YT));
    PetscCall(MatDestroy(&lbfgs->StY));
    PetscCall(MatDestroy(&lbfgs->L));
    PetscCall(MatDestroy(&lbfgs->D));
    PetscCall(MatDestroy(&lbfgs->R));
    PetscCall(MatDestroy(&lbfgs->Rinv));
    PetscCall(MatDestroy(&lbfgs->STfull));
    PetscCall(MatDestroy(&lbfgs->YTfull));
    PetscCall(MatDestroy(&lbfgs->StYfull));
    PetscCall(MatDestroy(&lbfgs->Lfull));
    PetscCall(MatDestroy(&lbfgs->Dfull));
    PetscCall(MatDestroy(&lbfgs->Rfull));
    PetscCall(VecDestroy(&lbfgs->rwork1));
    PetscCall(VecDestroy(&lbfgs->rwork2));
    PetscCall(VecDestroy(&lbfgs->rwork3));
    PetscCall(VecDestroy(&lbfgs->rwork4));
    PetscCall(VecDestroy(&lbfgs->lwork1));
    PetscCall(VecDestroy(&lbfgs->lwork2));
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
      //PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)B), lmvm->m, n, lmvm->m, N, n, NULL, N, NULL, &lbfgs->STfull));
      //PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)B), lmvm->m, n, lmvm->m, N, n, NULL, N, NULL, &lbfgs->YTfull));
      //for (i=0; i<lmvm->m; i++) {
      //  for (j=0; j<N; j++) {
      //    PetscCall(MatSetValue(lbfgs->STfull, i, j, 1.0, INSERT_VALUES));
      //    PetscCall(MatSetValue(lbfgs->YTfull, i, j, 1.0, INSERT_VALUES));
      //  }
      //}
      //PetscCall(MatAssemblyBegin(lbfgs->STfull, MAT_FINAL_ASSEMBLY));
      //PetscCall(MatAssemblyEnd(lbfgs->STfull, MAT_FINAL_ASSEMBLY));
      //PetscCall(MatAssemblyBegin(lbfgs->YTfull, MAT_FINAL_ASSEMBLY));
      //PetscCall(MatAssemblyEnd(lbfgs->YTfull, MAT_FINAL_ASSEMBLY));

      PetscCall(MatCreateDenseMatchingVec(X, lmvm->m, n, lmvm->m, N, NULL, &lbfgs->STfull));
      PetscCall(MatDuplicate(lbfgs->STfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->YTfull));
      /* Dense formats also do not fully support some of the Mat tools being used in this implementation */
      /* Create intermediate (sequential and small) matrices */
      PetscCall(MatMatTransposeMult(lbfgs->STfull, lbfgs->YTfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Lfull));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Dfull));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Rfull));
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork1));
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork2));
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
    PetscCall(MatDestroy(&lbfgs->ST));
    PetscCall(MatDestroy(&lbfgs->YT));
    PetscCall(MatDestroy(&lbfgs->StY));
    PetscCall(MatDestroy(&lbfgs->L));
    PetscCall(MatDestroy(&lbfgs->D));
    PetscCall(MatDestroy(&lbfgs->R));
    PetscCall(MatDestroy(&lbfgs->Rinv));
    PetscCall(MatDestroy(&lbfgs->STfull));
    PetscCall(MatDestroy(&lbfgs->YTfull));
    PetscCall(MatDestroy(&lbfgs->StYfull));
    PetscCall(MatDestroy(&lbfgs->Lfull));
    PetscCall(MatDestroy(&lbfgs->Dfull));
    PetscCall(MatDestroy(&lbfgs->Rfull));
    PetscCall(VecDestroy(&lbfgs->rwork1));
    PetscCall(VecDestroy(&lbfgs->rwork2));
    PetscCall(VecDestroy(&lbfgs->rwork3));
    PetscCall(VecDestroy(&lbfgs->rwork4));
    PetscCall(VecDestroy(&lbfgs->lwork1));
    PetscCall(VecDestroy(&lbfgs->lwork2));
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
