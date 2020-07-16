#include "sindy_impl.h"

PetscClassId  SPARSEREG_CLASSID;
PetscLogEvent SparseReg_STLSQ;
PetscLogEvent SparseReg_RLS;
PetscLogEvent SparseReg_LS;

PetscErrorCode SparseRegCreate(SparseReg* new_sparse_reg)
{
  PetscErrorCode  ierr;
  SparseReg sparse_reg;

  PetscFunctionBegin;
  PetscValidPointer(new_sparse_reg,1);
  *new_sparse_reg = NULL;
  ierr = SparseRegInitializePackage();CHKERRQ(ierr);

  ierr = PetscMalloc1(1, &sparse_reg);CHKERRQ(ierr);
  sparse_reg->threshold = 1e-5;
  sparse_reg->iterations = 10;
  sparse_reg->monitor = PETSC_FALSE;
  sparse_reg->use_regularization = PETSC_TRUE;
  sparse_reg->solve_normal = PETSC_FALSE;
  sparse_reg->solver_iterations = 0;

  *new_sparse_reg = sparse_reg;
  PetscFunctionReturn(0);
}

PetscErrorCode SparseRegDestroy(SparseReg* sparse_reg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sparse_reg) PetscFunctionReturn(0);
  ierr = PetscFree(*sparse_reg);CHKERRQ(ierr);
  *sparse_reg = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SparseRegSetThreshold(SparseReg sparse_reg, PetscReal threshold)
{
  PetscFunctionBegin;
  sparse_reg->threshold = threshold;
  PetscFunctionReturn(0);
}

PetscErrorCode SparseRegSetMonitor(SparseReg sparse_reg, PetscBool monitor)
{
  PetscFunctionBegin;
  sparse_reg->monitor = monitor;
  PetscFunctionReturn(0);
}

PetscErrorCode SparseRegGetTotalIterationNumber(SparseReg sparse_reg, PetscInt* iter)
{
  PetscFunctionBegin;
  *iter = sparse_reg->solver_iterations;
  PetscFunctionReturn(0);
}


PetscErrorCode SparseRegSetFromOptions(SparseReg sparse_reg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"sparse_reg_","Sparse regression options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-threshold","coefficients below this are set to 0","",sparse_reg->threshold,&sparse_reg->threshold,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iterations","number of thresholded iterations to do","",sparse_reg->iterations,&sparse_reg->iterations,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-monitor","print out additional information","",sparse_reg->monitor,&sparse_reg->monitor,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_regularization","whether to use regularization or not","",sparse_reg->use_regularization,&sparse_reg->use_regularization,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-solve_normal","for KSP, solve normal equations instead of original LS problem","",sparse_reg->solve_normal,&sparse_reg->solve_normal,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Sequential thresholded least squares. */
PetscErrorCode SparseRegSTLSQR(SparseReg sparse_reg, Mat A, Vec b, Mat D, Vec X)
{
  PetscErrorCode    ierr;
  PetscInt          i,k,j,R,C;
  PetscInt          old_num_thresholded,num_thresholded;
  PetscBool         *mask;
  PetscReal         *x,*zeros;
  Mat               A_thresh;
  PetscInt          *idR, *idC_thresh;
  Vec               *null_vecs, bounds[2];
  MatNullSpace      nullsp;
  const PetscScalar one = 1;

  PetscFunctionBegin;
  PetscLogEventBegin(SparseReg_STLSQ,0,0,0,0);
  if (sparse_reg->use_regularization) {
    ierr = SparseRegRLS(sparse_reg, A, b, NULL, NULL, D, X);CHKERRQ(ierr);
  } else {
    ierr = SparseRegLS(sparse_reg, A, b, X);CHKERRQ(ierr);
  }

  if (sparse_reg->threshold <= 0) {
    PetscLogEventEnd(SparseReg_STLSQ,0,0,0,0);
    PetscFunctionReturn(0);
  }

  ierr = MatGetSize(A, &R, &C);CHKERRQ(ierr);

  /* Create a workspace for thresholding. */
  ierr = MatDuplicate(A, MAT_COPY_VALUES, &A_thresh);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(X, C, &null_vecs);CHKERRQ(ierr);
  ierr = VecDuplicate(X, &bounds[0]);CHKERRQ(ierr);
  ierr = VecDuplicate(X, &bounds[1]);CHKERRQ(ierr);

  ierr = VecSet(bounds[0], PETSC_NINFINITY);CHKERRQ(ierr);
  ierr = VecSet(bounds[1], PETSC_INFINITY);CHKERRQ(ierr);

  ierr = PetscCalloc1(R, &zeros);CHKERRQ(ierr);
  ierr = PetscMalloc3(C, &mask, R, &idR, C, &idC_thresh);CHKERRQ(ierr);
  for (i=0;i<R;i++) idR[i] = i;
  for (i=0;i<C;i++) mask[i] = PETSC_FALSE;

  /* Repeatedly threshold and perform least squares on the non-thresholded values. */
  num_thresholded = 0;
  ierr = MatCopy(A, A_thresh, SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  for (k = 0; k < sparse_reg->iterations; k++) {
    /* Threshold the data. */
    old_num_thresholded = num_thresholded;
    ierr = VecGetArray(X, &x);CHKERRQ(ierr);
    for (j = 0; j < C; j++) {
      if (!mask[j]) {
        if (PetscAbsReal(x[j]) < sparse_reg->threshold) {
          x[j] = 0;
          mask[j] = PETSC_TRUE;
          idC_thresh[num_thresholded] = j;

          /* Add null vector for this column. */
          ierr = VecZeroEntries(null_vecs[num_thresholded]);CHKERRQ(ierr);
          ierr = VecSetValues(null_vecs[num_thresholded], 1, &j, &one, INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecAssemblyBegin(null_vecs[num_thresholded]);CHKERRQ(ierr);
          ierr = VecAssemblyEnd(null_vecs[num_thresholded]);CHKERRQ(ierr);

          /* Set bounds for this object. */
          ierr = VecSetValue(bounds[0], j, 0, INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecSetValue(bounds[1], j, 0, INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecAssemblyBegin(bounds[0]);CHKERRQ(ierr);
          ierr = VecAssemblyBegin(bounds[1]);CHKERRQ(ierr);
          ierr = VecAssemblyEnd(bounds[0]);CHKERRQ(ierr);
          ierr = VecAssemblyEnd(bounds[1]);CHKERRQ(ierr);


          num_thresholded++;
        }
      } else {
          x[j] = 0;
      }
    }
    ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
    if (sparse_reg->monitor) {
      PetscPrintf(PETSC_COMM_SELF, "SparseReg: iteration: %d, nonzeros: %d\n", k, C - num_thresholded);
    }
    if (old_num_thresholded == num_thresholded) break;

    /* Run sparse least squares on the non-zero basis functions. */
    if (sparse_reg->use_regularization) {
      ierr = SparseRegRLS(sparse_reg, A_thresh, b, bounds[0], bounds[1], D, X);CHKERRQ(ierr);
    } else {
      /* Zero out the newly thresholded columns of the matrix. */
      for (j = old_num_thresholded; j < num_thresholded; j++) {
        ierr = MatSetValues(A_thresh,R,idR,1,idC_thresh+j,zeros,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      /* Record this as the null space of the matrix. */
      ierr = MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, num_thresholded, &null_vecs[0], &nullsp);CHKERRQ(ierr);
      ierr = MatSetNullSpace(A_thresh, nullsp);CHKERRQ(ierr);

      ierr = SparseRegLS(sparse_reg, A_thresh, b, X);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
    }
  }

  if (sparse_reg->use_regularization) {
    /* Zero out the thresholded columns of the matrix. */
    for (j = 0; j < num_thresholded; j++) {
      ierr = MatSetValues(A_thresh,R,idR,1,idC_thresh+j,zeros,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Record this as the null space of the matrix. */
    ierr = MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, num_thresholded, &null_vecs[0], &nullsp);CHKERRQ(ierr);
    ierr = MatSetNullSpace(A_thresh, nullsp);CHKERRQ(ierr);

    /* Do an unregularized regression to get the final values. */
    ierr = SparseRegLS(sparse_reg, A_thresh, b, X);CHKERRQ(ierr);
  }

  /* Maybe I should zero out the thresholded entries again here, just to make sure Tao didn't mess them up. */
  ierr = VecGetArray(X, &x);CHKERRQ(ierr);
  for (j = 0; j < num_thresholded; j++) {
    x[idC_thresh[j]] = 0;
  }
  ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);

  ierr = PetscFree(zeros);CHKERRQ(ierr);
  ierr = PetscFree3(mask, idR, idC_thresh);CHKERRQ(ierr);
  ierr = VecDestroyVecs(C, &null_vecs);CHKERRQ(ierr);
  ierr = VecDestroy(&bounds[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&bounds[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&A_thresh);CHKERRQ(ierr);
  PetscLogEventEnd(SparseReg_STLSQ,0,0,0,0);
  PetscFunctionReturn(0);
}

typedef struct {
  Mat       A,D;
  Vec       b;
  PetscInt  M,N,K;
} LeastSquaresCtx;

PetscErrorCode InitializeLeastSquaresData(LeastSquaresCtx *ctx, Mat A, Mat D, Vec b)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ctx->A = A;
  ctx->D = D;
  ctx->b = b;
  ierr = MatGetSize(ctx->A,&ctx->M,&ctx->N);CHKERRQ(ierr);
  if (D) {
    ierr = MatGetSize(ctx->D,&ctx->K,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  LeastSquaresCtx *ctx = (LeastSquaresCtx *)ptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatMult(ctx->A, X, F);CHKERRQ(ierr);
  ierr = VecAXPY(F, -1.0, ctx->b);CHKERRQ(ierr);
  PetscLogFlops(ctx->M*ctx->N*2);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ptr)
{
  /* The Jacobian is constant for this problem. */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode FormStartingPoint(Vec X)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Find x to minimize ||Ax - b||_2  where A is an m x n matrix. */
PetscErrorCode SparseRegLS(SparseReg sparse_reg, Mat A, Vec b, Vec x)
{
  PetscErrorCode  ierr;
  Vec             Ab;
  Mat             N;
  KSP             ksp;

  PetscFunctionBegin;
  PetscLogEventBegin(SparseReg_LS,0,0,0,0);

  ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);

  ierr = MatCreateNormal(A,&N);CHKERRQ(ierr);
  if (sparse_reg->solve_normal) {
    ierr = VecDuplicate(x,&Ab);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,b,Ab);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,N,N);CHKERRQ(ierr);
  } else {
    PC pc;
    ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,N);CHKERRQ(ierr);
  }
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  if (sparse_reg->solve_normal) {
    ierr = KSPSolve(ksp,Ab,x);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  }
  sparse_reg->solver_iterations += 1;

  if (sparse_reg->solve_normal) {
    ierr = VecDestroy(&Ab);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&N);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  PetscLogEventEnd(SparseReg_LS,0,0,0,0);
  PetscFunctionReturn(0);
}

/* Find x to minimize ||Ax - b||_2 + ||Dx||_1 where A is an m x n matrix. If D is
   null, it will default to the identity. */
PetscErrorCode SparseRegRLS(SparseReg sparse_reg, Mat A, Vec b, Vec LB, Vec UB, Mat D, Vec x)
{
  PetscErrorCode  ierr;
  Vec             f;               /* solution, function f(x) = A*x-b */
  Mat             J;               /* Jacobian matrix, Transform matrix */
  Tao             tao;                /* Tao solver context */
  LeastSquaresCtx ctx;
  PetscBool       flg;
  PetscInt        iter;

  PetscFunctionBegin;
  PetscLogEventBegin(SparseReg_RLS,0,0,0,0);
  ierr = InitializeLeastSquaresData(&ctx, A, D, b);CHKERRQ(ierr);
  J = A;

  /* Allocate vector function vector */
  ierr = VecDuplicate(b, &f);CHKERRQ(ierr);

  /* Use l1dict by default. */
  ierr = PetscOptionsHasName(NULL, NULL, "-tao_brgn_regularization_type", &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscOptionsSetValue(NULL, "-tao_brgn_regularization_type", "l1dict");CHKERRQ(ierr);
  }

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBRGN);CHKERRQ(ierr);

  /* Set bounds. */
  ierr = TaoSetVariableBounds(tao, LB, UB);CHKERRQ(ierr);

  /* Set initial guess */
  ierr = FormStartingPoint(x);CHKERRQ(ierr);

  /* Bind x to tao->solution. */
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);

  /* Bind D to tao->data->D */
  ierr = TaoBRGNSetDictionaryMatrix(tao,D);CHKERRQ(ierr);

  /* Set the function and Jacobian routines. */
  ierr = TaoSetResidualRoutine(tao,f,EvaluateFunction,(void*)&ctx);CHKERRQ(ierr);
  ierr = TaoSetJacobianResidualRoutine(tao,J,J,EvaluateJacobian,(void*)&ctx);CHKERRQ(ierr);

  /* Check for any TAO command line arguments */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);
  ierr = TaoGetTotalIterationNumber(tao, &iter);CHKERRQ(ierr);
  sparse_reg->solver_iterations += iter;

   /* Free PETSc data structures */
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscLogEventEnd(SparseReg_RLS,0,0,0,0);
  PetscFunctionReturn(0);
}

