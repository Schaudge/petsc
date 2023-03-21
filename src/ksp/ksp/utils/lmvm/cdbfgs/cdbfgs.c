#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS       *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  } else {
    ierr = MatMult(lbfgs->diag_bfgs, X, Z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS       *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  } else {
    ierr = MatSolve(lbfgs->diag_bfgs, F, dX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  /* Start with the H0 term */
  ierr = MatCDBFGSApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  if (lmvm->k == -1) {
    PetscFunctionReturn(0); /* No updates stored yet */
  }

  /* Apply the Phi^T = [Y^TH; S^T] to the RHS vector F */
  /* The result is stored in two halves, (rwork1 = Y^T H F) and (rwork2 = S^T F) */
  ierr = MatCDBFGSApplyJ0Inv(B, F, lbfgs->lwork1);CHKERRQ(ierr);
  ierr = MatMult(lbfgs->YT, lbfgs->lwork1, lbfgs->rwork1);CHKERRQ(ierr);
  ierr = MatMult(lbfgs->ST, F, lbfgs->rwork2);CHKERRQ(ierr);

  /* Calculate dX = HY R^{-T) rwork2 */
  /* This concludes operations with top half of M */
  ierr = MatSolveTranspose(lbfgs->Rinv, lbfgs->rwork2, lbfgs->rwork3);CHKERRQ(ierr);
  ierr = VecScale(lbfgs->rwork3, -1.0);CHKERRQ(ierr);
  ierr = MatMultTranspose(lbfgs->YT, lbfgs->rwork3, lbfgs->lwork1);CHKERRQ(ierr);
  ierr = MatCDBFGSApplyJ0Inv(B, lbfgs->lwork1, lbfgs->lwork2);CHKERRQ(ierr);
  ierr = VecAXPY(dX, 1.0, lbfgs->lwork2);CHKERRQ(ierr);

  /* Calculate rwork3 = -R^{-T} rwork1 */
  ierr = MatSolveTranspose(lbfgs->Rinv, lbfgs->rwork1, lbfgs->rwork3);CHKERRQ(ierr);
  ierr = VecScale(lbfgs->rwork3, -1.0);CHKERRQ(ierr);

  /* Calculate rwork3 += R^{-T}(D + YtHY)R^{-1} rwork2 */
  ierr = MatSolve(lbfgs->Rinv, lbfgs->rwork2, lbfgs->rwork4);CHKERRQ(ierr);
  ierr = MatMultTranspose(lbfgs->YT, lbfgs->rwork4, lbfgs->lwork1);CHKERRQ(ierr);
  ierr = MatCDBFGSApplyJ0Inv(B, lbfgs->lwork1, lbfgs->lwork2);CHKERRQ(ierr);
  ierr = MatMult(lbfgs->YT, lbfgs->lwork2, lbfgs->rwork2);CHKERRQ(ierr);
  ierr = MatMultAdd(lbfgs->D, lbfgs->rwork4, lbfgs->rwork2, lbfgs->rwork2);CHKERRQ(ierr);
  ierr = MatSolveTransposeAdd(lbfgs->Rinv, lbfgs->rwork2, lbfgs->rwork3, lbfgs->rwork3);CHKERRQ(ierr);
  
  /* Calculate dX += S rwork3 */
  /* This concludes operations with bottom half of M */
  ierr = MatMultTransposeAdd(lbfgs->ST, lbfgs->rwork3, dX, dX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMCDBFGS(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Forward product not yet implemented");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *dctx;
  PetscErrorCode    ierr;
  const PetscScalar *xx, *ff, *vals;
  PetscScalar       *buffer, curvature, ststmp;
  PetscReal         curvtol;
  const PetscInt    *cols;
  PetscInt          n, low, high, *rows, i, j;
  IS                active_rows, perm, iperm;
  MatFactorInfo     info;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp);CHKERRQ(ierr);
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
        ierr = PetscMalloc1(1, &rows);CHKERRQ(ierr);
        ierr = PetscMalloc1(B->rmap->n, &buffer);CHKERRQ(ierr);
        for (i=0; i<lmvm->k; i++) {
          rows[0] = i+1;
          /* Take the information one row ahead of the current idx */
          ierr = MatGetRow(lbfgs->STfull, i+1, &n, &cols, &vals);CHKERRQ(ierr);
          /* Copy the info into a buffer array and restore the row */
          for (j=0; j<n; j++) buffer[j] = vals[j];
          ierr = MatRestoreRow(lbfgs->STfull, i+1, &n, &cols, &vals);CHKERRQ(ierr);
          /* Place the info from the next row into this one, overwriting existing info */
          /* This process ultimately discards the information stored in the first row at idx 0 */
          /* New information can then be written into idx=lmvm->k */
          ierr = MatSetValues(lbfgs->STfull, 1, rows, n, cols, buffer, INSERT_VALUES);CHKERRQ(ierr);
          /* Repeat for the Y matrix */
          ierr = MatGetRow(lbfgs->YTfull, i+1, &n, &cols, &vals);CHKERRQ(ierr);
          for (j=0; j<n; j++) buffer[j] = vals[j];
          ierr = MatRestoreRow(lbfgs->YTfull, i+1, &n, &cols, &vals);CHKERRQ(ierr);
          ierr = MatSetValues(lbfgs->YTfull, 1, rows, n, cols, buffer, INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = PetscFree(rows);CHKERRQ(ierr);
        ierr = PetscFree(buffer);CHKERRQ(ierr);
      } else {
        lmvm->k = lmvm->k + 1;
      }
      /* Generate the required row/col idx arrays for data transfer */
      ierr = VecGetLocalSize(lmvm->Xprev, &n);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(lmvm->Xprev, &low, &high);CHKERRQ(ierr);
      ierr = PetscMalloc2(1, &lbfgs->idx_rows, n, &lbfgs->idx_cols);CHKERRQ(ierr);
      lbfgs->idx_rows[0] = lmvm->k;
      for (i=low; i<high; i++) {
        lbfgs->idx_cols[i] = i;
      }
      /* First update the S^T matrix */
      ierr = VecGetArrayRead(lmvm->Xprev, &xx);CHKERRQ(ierr);
      ierr = MatSetValues(lbfgs->STfull, 1, lbfgs->idx_rows, n, lbfgs->idx_cols, xx, INSERT_VALUES);
      ierr = MatAssemblyBegin(lbfgs->STfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(lbfgs->STfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(lmvm->Xprev, &xx);CHKERRQ(ierr);
      /* Now repeat update for the Y^T matrix */
      ierr = VecGetArrayRead(lmvm->Fprev, &ff);CHKERRQ(ierr);
      ierr = MatSetValues(lbfgs->YTfull, 1, lbfgs->idx_rows, n, lbfgs->idx_cols, ff, INSERT_VALUES);
      ierr = MatAssemblyBegin(lbfgs->YTfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(lbfgs->YTfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(lmvm->Xprev, &ff);CHKERRQ(ierr);
      /* Clean up unnecessary arrays */
      ierr = PetscFree2(lbfgs->idx_rows, lbfgs->idx_cols);CHKERRQ(ierr);
      /* Create and fill the intermediate matrices */
      ierr = MatDestroy(&lbfgs->StYfull);CHKERRQ(ierr);
      ierr = MatMatTransposeMult(lbfgs->STfull, lbfgs->YTfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull);CHKERRQ(ierr);
      ierr = MatConvert(lbfgs->StYfull, lbfgs->dense_type, MAT_INPLACE_MATRIX, &lbfgs->StYfull);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->Lfull);CHKERRQ(ierr);
      ierr = MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Lfull);CHKERRQ(ierr);
      ierr = MatZeroEntries(lbfgs->Lfull);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->Dfull);CHKERRQ(ierr);
      ierr = MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Dfull);CHKERRQ(ierr);
      ierr = MatZeroEntries(lbfgs->Dfull);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->Rfull);CHKERRQ(ierr);
      ierr = MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Rfull);CHKERRQ(ierr);
      ierr = MatZeroEntries(lbfgs->Rfull);CHKERRQ(ierr);
      for (i=0; i<lmvm->m; i++) {
        ierr = MatGetRow(lbfgs->StYfull, i, &n, NULL, &vals);CHKERRQ(ierr);
        for (j=0; j<n; j++) {
          if (i <= j) {
            ierr = MatSetValue(lbfgs->Rfull, i, j, vals[j], INSERT_VALUES);
            if (i == j) {
              ierr = MatSetValue(lbfgs->Dfull, i, j, vals[j], INSERT_VALUES);
            }
          } else {
            ierr = MatSetValue(lbfgs->Lfull, i, j, vals[j], INSERT_VALUES);
          }
        }
        ierr = MatRestoreRow(lbfgs->StYfull, i, &n, NULL, &vals);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(lbfgs->Lfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(lbfgs->Lfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(lbfgs->Dfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(lbfgs->Dfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(lbfgs->Rfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(lbfgs->Rfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      /* Clear out the previously formed submatrices and work vectors */
      ierr = MatDestroy(&lbfgs->ST);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->YT);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->StY);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->L);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->D);CHKERRQ(ierr);
      ierr = MatDestroy(&lbfgs->R);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->rwork1);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->rwork2);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->rwork3);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->rwork4);CHKERRQ(ierr);
      if (lmvm->k == lmvm->m-1) {
        /* At maximum storage so the submatrices are equal to the full matrices */
        lbfgs->ST = lbfgs->STfull;
        ierr = PetscObjectReference((PetscObject)lbfgs->STfull);CHKERRQ(ierr);
        lbfgs->YT = lbfgs->YTfull;
        ierr = PetscObjectReference((PetscObject)lbfgs->YTfull);CHKERRQ(ierr);
        lbfgs->StY = lbfgs->StYfull;
        ierr = PetscObjectReference((PetscObject)lbfgs->StYfull);CHKERRQ(ierr);
        lbfgs->L = lbfgs->Lfull;
        ierr = PetscObjectReference((PetscObject)lbfgs->Lfull);CHKERRQ(ierr);
        lbfgs->D = lbfgs->Dfull;
        ierr = PetscObjectReference((PetscObject)lbfgs->Dfull);CHKERRQ(ierr);
        lbfgs->R = lbfgs->Rfull;
        ierr = PetscObjectReference((PetscObject)lbfgs->Rfull);CHKERRQ(ierr);
      } else {
        /* There's unstored rows of ST and YT so we have to generate submatrices */
        ierr = ISCreateStride(comm, lmvm->k+1, 0, 1, &active_rows);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(lbfgs->STfull, active_rows, NULL, MAT_INITIAL_MATRIX, &lbfgs->ST);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(lbfgs->YTfull, active_rows, NULL, MAT_INITIAL_MATRIX, &lbfgs->YT);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(lbfgs->StYfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->StY);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(lbfgs->Lfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->L);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(lbfgs->Dfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->D);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(lbfgs->Rfull, active_rows, active_rows, MAT_INITIAL_MATRIX, &lbfgs->R);CHKERRQ(ierr);
        ierr = ISDestroy(&active_rows);CHKERRQ(ierr);
      }
      /* Generate the work vectors from the submatrices */
      ierr = MatCreateVecs(lbfgs->R, &lbfgs->rwork1, &lbfgs->rwork2);CHKERRQ(ierr);
      ierr = MatCreateVecs(lbfgs->R, &lbfgs->rwork3, &lbfgs->rwork4);CHKERRQ(ierr);
      /* Factor the R matrix for inversion */
      ierr = MatDestroy(&lbfgs->Rinv);CHKERRQ(ierr);
      ierr = MatDuplicate(lbfgs->R, MAT_COPY_VALUES, &lbfgs->Rinv);CHKERRQ(ierr);
      ierr = MatGetOrdering(lbfgs->Rinv, MATORDERINGRCM, &perm, &iperm);CHKERRQ(ierr);
      ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
      info.fill = 0.0;
      info.dtcol = 0.0;
      info.zeropivot = 1e-12;
      info.pivotinblocks = 0.0;
      ierr = MatLUFactor(lbfgs->Rinv, perm, iperm, &info);
      /* Update the diagonal H0 if it exists */
      if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
        ierr = MatLMVMUpdate(lbfgs->diag_bfgs, X, F);CHKERRQ(ierr);
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
      ierr = VecSet(dctx->invD, lbfgs->delta);CHKERRQ(ierr);
    }
  }
  
  if (lbfgs->watchdog > lbfgs->max_seq_rejects) {
    ierr = MatLMVMReset(B, PETSC_FALSE);CHKERRQ(ierr);
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      ierr = MatLMVMReset(lbfgs->diag_bfgs, PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMCDBFGS(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *blbfgs = (Mat_CDBFGS*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_CDBFGS        *mlbfgs = (Mat_CDBFGS*)mdata->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  mlbfgs->watchdog        = blbfgs->watchdog;
  mlbfgs->max_seq_rejects = blbfgs->max_seq_rejects;
  if (!(bdata->J0 || bdata->user_pc || bdata->user_ksp || bdata->user_scale)) {
    ierr = MatCopy(blbfgs->diag_bfgs, mlbfgs->diag_bfgs, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMCDBFGS(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  lbfgs->watchdog = 0;
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    ierr = MatLMVMReset(lbfgs->diag_bfgs, destructive);CHKERRQ(ierr);
  }
  if (lbfgs->allocated && destructive) {
    ierr = MatDestroy(&lbfgs->STfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->YTfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->StYfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Lfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Dfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Rfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->ST);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->YT);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->StY);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->L);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->D);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->R);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Rinv);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork1);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork2);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork3);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork4);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->lwork1);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->lwork2);CHKERRQ(ierr);
    lbfgs->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscBool         same, allocate = PETSC_FALSE;
  VecType           vec_type;
  PetscInt          m, n, M, N, i, j;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  if (lmvm->allocated) {
    ierr = VecGetType(X, &vec_type);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)lmvm->Xprev, vec_type, &same);CHKERRQ(ierr);
    if (!same) {
      /* Given X vector has a different type than allocated X-type data structures.
         We need to destroy all of this and duplicate again out of the given vector. */
      allocate = PETSC_TRUE;
      ierr = MatLMVMReset(B, PETSC_TRUE);CHKERRQ(ierr);
    } else {
      VecCheckMatCompatible(B, X, 2, F, 3);
    }
  } else {
    allocate = PETSC_TRUE;
  }
  if (allocate) {
    ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
    ierr = VecGetSize(X, &N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(F, &m);CHKERRQ(ierr);
    ierr = VecGetSize(F, &M);CHKERRQ(ierr);
    if (N != M) SETERRQ(comm, PETSC_ERR_ARG_SIZ, "Incorrect problem sizes! dim(X) not equal to dim(F)");
    ierr = MatSetSizes(B, m, n, M, N);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecDuplicate(F, &lmvm->Fprev);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = PetscObjectBaseTypeCompare((PetscObject)X, VECCUDA, &same);CHKERRQ(ierr);
      if (same) {
        lbfgs->dense_type = MATSEQDENSECUDA;
        ierr = MatCreateAIJCUSPARSE(PetscObjectComm((PetscObject)B), lmvm->m, n, lmvm->m, N, n, NULL, N, NULL, &lbfgs->STfull);CHKERRQ(ierr);
        ierr = MatCreateAIJCUSPARSE(PetscObjectComm((PetscObject)B), lmvm->m, n, lmvm->m, N, n, NULL, N, NULL, &lbfgs->YTfull);CHKERRQ(ierr);
      } else {
        lbfgs->dense_type = MATSEQDENSE;
        ierr = MatCreateAIJ(PetscObjectComm((PetscObject)B), lmvm->m, n, lmvm->m, N, n, NULL, N, NULL, &lbfgs->STfull);CHKERRQ(ierr);
        ierr = MatCreateAIJ(PetscObjectComm((PetscObject)B), lmvm->m, n, lmvm->m, N, n, NULL, N, NULL, &lbfgs->YTfull);CHKERRQ(ierr);
      }
      for (i=0; i<lmvm->m; i++) {
        for (j=0; j<N; j++) {
          ierr = MatSetValue(lbfgs->STfull, i, j, 1.0, INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValue(lbfgs->YTfull, i, j, 1.0, INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      ierr = MatAssemblyBegin(lbfgs->STfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(lbfgs->STfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(lbfgs->YTfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(lbfgs->YTfull, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = VecDuplicate(lmvm->Xprev, &lbfgs->lwork1);
    ierr = VecDuplicate(lmvm->Xprev, &lbfgs->lwork2);
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      ierr = MatLMVMAllocate(lbfgs->diag_bfgs, X, F);CHKERRQ(ierr);
    }
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMCDBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbfgs->allocated) {
    ierr = MatDestroy(&lbfgs->STfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->YTfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->StYfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Lfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Dfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Rfull);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->ST);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->YT);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->StY);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->L);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->D);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->R);CHKERRQ(ierr);
    ierr = MatDestroy(&lbfgs->Rinv);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork1);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork2);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork3);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->rwork4);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->lwork1);CHKERRQ(ierr);
    ierr = VecDestroy(&lbfgs->lwork2);CHKERRQ(ierr);
    lbfgs->allocated = PETSC_FALSE;
  }
  ierr = MatDestroy(&lbfgs->D);CHKERRQ(ierr);
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMCDBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscInt          m, n, M, N;
  PetscMPIInt       size;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  Vec               Xtmp, Ftmp;

  PetscFunctionBegin;
  ierr = MatGetSize(B, &M, &N);CHKERRQ(ierr);
  if (M == 0 && N == 0) SETERRQ(comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = VecCreateSeq(comm, N, &Xtmp);CHKERRQ(ierr);
      ierr = VecCreateSeq(comm, M, &Ftmp);CHKERRQ(ierr);
    } else {
      ierr = MatGetLocalSize(B, &m, &n);CHKERRQ(ierr);
      ierr = VecCreateMPI(comm, n, N, &Xtmp);CHKERRQ(ierr);
      ierr = VecCreateMPI(comm, m, M, &Ftmp);CHKERRQ(ierr);
    }
    ierr = MatAllocate_LMVMCDBFGS(B, Xtmp, Ftmp);CHKERRQ(ierr);
    ierr = VecDestroy(&Xtmp);CHKERRQ(ierr);
    ierr = VecDestroy(&Ftmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVMCDBFGS(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscBool         isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = MatView_LMVM(B, pv);CHKERRQ(ierr);
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    ierr = MatView(lbfgs->diag_bfgs, pv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMCDBFGS(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_CDBFGS        *lbfgs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMCDBFGS);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->view = MatView_LMVMCDBFGS;
  B->ops->setup = MatSetUp_LMVMCDBFGS;
  B->ops->destroy = MatDestroy_LMVMCDBFGS;
  B->ops->solve = MatSolve_LMVMCDBFGS;
  
  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMCDBFGS;
  lmvm->ops->reset = MatReset_LMVMCDBFGS;
  lmvm->ops->update = MatUpdate_LMVMCDBFGS;
  lmvm->ops->mult = MatMult_LMVMCDBFGS;
  lmvm->ops->copy = MatCopy_LMVMCDBFGS;
  
  ierr = PetscNewLog(B, &lbfgs);CHKERRQ(ierr);
  lmvm->ctx = (void*)lbfgs;
  lbfgs->allocated       = PETSC_FALSE;
  lbfgs->watchdog        = 0;
  lbfgs->delta           = 1.0;
  lbfgs->delta_min       = 1e-7;
  lbfgs->delta_max       = 100.0;
  lbfgs->max_seq_rejects = lmvm->m/2;
  
  ierr = MatCreate(PetscObjectComm((PetscObject)B), &lbfgs->diag_bfgs);CHKERRQ(ierr);
  ierr = MatSetType(lbfgs->diag_bfgs, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(lbfgs->diag_bfgs, "J0_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMCDBFGS);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}