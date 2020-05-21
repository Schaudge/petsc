#include "sindy.h"

PetscErrorCode FormStartingPoint(Vec);
PetscErrorCode EvaluateFunction(Tao,Vec,Vec,void *);
PetscErrorCode EvaluateJacobian(Tao,Vec,Mat,Mat,void *);

PetscInt SINDyCountBases(PetscInt poly_order, PetscInt sine_order)
{
  return poly_order + 1 + 2 * sine_order;
}

PetscErrorCode SINDyCreateBasis(Vec x, PetscInt poly_order, PetscInt sine_order, Mat* Theta, PetscInt *num_bases)
{
  PetscErrorCode   ierr;
  PetscInt         x_size;
  const PetscReal* x_data;
  PetscInt         B, i, r, o;
  PetscReal        *Theta_data;
  PetscInt         *idx, *idb;

  PetscFunctionBegin;
  ierr = VecGetSize(x, &x_size);CHKERRQ(ierr);

  B = SINDyCountBases(poly_order, sine_order);
  ierr = PetscMalloc1(B*x_size, &Theta_data);CHKERRQ(ierr);

  ierr = VecGetArrayRead(x, &x_data);CHKERRQ(ierr);
  i = 0;
  for (r = 0; r < x_size; r++) {
    for (o = 0; o <= poly_order; o++) {
      Theta_data[i] = PetscPowRealInt(x_data[r], o);
      i++;
    }
    for (o = 1; o <= sine_order; o++) {
      Theta_data[i] = PetscSinReal(o * x_data[r]);
      i++;
      Theta_data[i] = PetscCosReal(o * x_data[r]);
      i++;
    }
  }
  ierr = VecRestoreArrayRead(x, &x_data);CHKERRQ(ierr);

  ierr = PetscMalloc2(x_size, &idx, B, &idb);CHKERRQ(ierr);
  for (i=0;i<x_size;i++) idx[i] = i;
  for (i=0;i<B;i++)      idb[i] = i;

  ierr = MatCreateSeqDense(PETSC_COMM_SELF, x_size, B, NULL, Theta);CHKERRQ(ierr);

  ierr = MatSetValues(*Theta,x_size,idx,B,idb,(PetscReal *)Theta_data,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(idx, idb);CHKERRQ(ierr);
  ierr = PetscFree(Theta_data);CHKERRQ(ierr);

  *num_bases = B;
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

/* Find x to minimize ||Ax - b||_2 + ||x||_1 where A is an m x n matrix. */
PetscErrorCode SINDySparseLeastSquares(Mat A, Vec b, Mat D, Vec x)
{
  PetscErrorCode ierr;
  Vec            f;               /* solution, function f(x) = A*x-b */
  Mat            J;               /* Jacobian matrix, Transform matrix */
  Tao            tao;                /* Tao solver context */
  PetscReal      hist[100],resid[100];
  PetscInt       lits[100];
  LeastSquaresCtx ctx;

  PetscFunctionBegin;
  ierr = InitializeLeastSquaresData(&ctx, A, D, b);CHKERRQ(ierr);
  J = A;

  /* Allocate vector function vector */
  ierr = VecDuplicate(b, &f);CHKERRQ(ierr);

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

  ierr = TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE);CHKERRQ(ierr);

  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  LeastSquaresCtx *ctx = (LeastSquaresCtx *)ptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatMult(ctx->A, X, F);CHKERRQ(ierr);
  ierr = VecAXPY(F, -1.0, ctx->b);CHKERRQ(ierr);
  PetscLogFlops(ctx->M*ctx->N*2);
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ptr)
{
  /* The Jacobian is constant for this problem. */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


PetscErrorCode FormStartingPoint(Vec X)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
