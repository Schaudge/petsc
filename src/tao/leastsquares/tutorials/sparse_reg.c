#include "sindy_impl.h"

static PetscErrorCode SparseLeastSquares(Mat A, Vec b, Mat D, Vec x);

PetscErrorCode SparseRegCreate(SparseReg* new_sparse_reg)
{
  PetscErrorCode  ierr;
  SparseReg sparse_reg;

  PetscFunctionBegin;
  PetscValidPointer(new_sparse_reg,3);
  *new_sparse_reg = NULL;

  ierr = PetscMalloc1(1, &sparse_reg);CHKERRQ(ierr);
  sparse_reg->threshold = 1e-5;
  sparse_reg->iterations = 10;
  sparse_reg->monitor = PETSC_FALSE;

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

PetscErrorCode SparseRegSetFromOptions(SparseReg sparse_reg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"sparse_reg_","Sparse regression options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-threshold","coefficients below this are set to 0","",sparse_reg->threshold,&sparse_reg->threshold,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iterations","number of thresholded iterations to do","",sparse_reg->iterations,&sparse_reg->iterations,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-monitor","print out additional information","",sparse_reg->monitor,&sparse_reg->monitor,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Sequential thresholded least squares. */
PetscErrorCode SparseRegSTLSQ(SparseReg sparse_reg, Mat A, Vec b, Mat D, Vec X)
{
  PetscErrorCode ierr;
  PetscInt       i,k,j,R,C;
  PetscInt       old_num_thresholded,num_thresholded;
  PetscBool      *mask;
  PetscReal      *x,*zeros;
  Mat            A_thresh;
  PetscInt       *idR, *idC_thresh;

  PetscFunctionBegin;
  ierr = SparseLeastSquares(A, b, D, X);CHKERRQ(ierr);

  if (sparse_reg->threshold <= 0) PetscFunctionReturn(0);

  /* Create a workspace for thresholding. */
  ierr = MatDuplicate(A, MAT_COPY_VALUES, &A_thresh);CHKERRQ(ierr);

  ierr = MatGetSize(A, &R, &C);CHKERRQ(ierr);
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
          idC_thresh[num_thresholded] = j;
          num_thresholded++;
          mask[j] = PETSC_TRUE;
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

    /* Zero out those columns of the matrix. */
    for (j = old_num_thresholded; j < num_thresholded; j++) {
      ierr = MatSetValues(A_thresh,R,idR,1,idC_thresh+j,zeros,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Run sparse least squares on the non-zero basis functions. */
    ierr = SparseLeastSquares(A_thresh, b, D, X);CHKERRQ(ierr);
  }

  /* Maybe I should zero out the thresholded entries again here, just to make sure Tao didn't mess them up. */
  ierr = VecGetArray(X, &x);CHKERRQ(ierr);
  for (j = 0; j < num_thresholded; j++) {
    x[idC_thresh[j]] = 0;
  }
  ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
  ierr = PetscFree(zeros);CHKERRQ(ierr);
  ierr = PetscFree3(mask, idR, idC_thresh);CHKERRQ(ierr);
  ierr = MatDestroy(&A_thresh);CHKERRQ(ierr);
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

/* Find x to minimize ||Ax - b||_2 + ||Dx||_1 where A is an m x n matrix. If D is
   null, it will default to the identity. The given mask tells which entries in
   x to leave out of the optimization. If null, all entries will be optimized. */
static PetscErrorCode SparseLeastSquares(Mat A, Vec b, Mat D, Vec x)
{
  PetscErrorCode  ierr;
  Vec             f;               /* solution, function f(x) = A*x-b */
  Mat             J;               /* Jacobian matrix, Transform matrix */
  Tao             tao;                /* Tao solver context */
  LeastSquaresCtx ctx;
  PetscBool       flg;

  PetscFunctionBegin;
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

   /* Free PETSc data structures */
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
