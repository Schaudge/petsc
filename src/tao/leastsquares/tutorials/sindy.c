#include <petsc/private/petscimpl.h>
#include "sindy.h"

typedef struct _p_Data {
  PetscInt B,N,dim;
  Mat Theta;
  char** names;
  char* names_data;
} Data;

struct _p_Basis {
    PetscInt poly_order,sine_order;
    Data data;
};

PetscErrorCode FormStartingPoint(Vec);
PetscErrorCode EvaluateFunction(Tao,Vec,Vec,void *);
PetscErrorCode EvaluateJacobian(Tao,Vec,Mat,Mat,void *);

static PetscInt SINDyCountBases(PetscInt dim, PetscInt poly_order, PetscInt sine_order)
{
  return dim * (poly_order + 1 + 2 * sine_order);
}

PETSC_EXTERN PetscErrorCode SINDyBasisDataGetSize(Basis basis, PetscInt *N, PetscInt *B)
{
  if (N) *N = basis->data.N;
  if (B) *B = basis->data.B;
  return(0);
}

PetscErrorCode SINDyCreateBasis(PetscInt poly_order, PetscInt sine_order, Basis* new_basis)
{
  PetscErrorCode  ierr;
  Basis basis;

  PetscFunctionBegin;
  PetscValidPointer(new_basis,2);
  *new_basis = NULL;

  ierr = PetscMalloc1(1, &basis);CHKERRQ(ierr);
  basis->poly_order = poly_order;
  basis->sine_order = sine_order;

  basis->data.B = 0;
  basis->data.N = 0;
  basis->data.dim = 0;
  basis->data.Theta = NULL;
  basis->data.names = NULL;
  basis->data.names_data = NULL;

  *new_basis = basis;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyBasisDestroy(Basis* basis)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*basis) PetscFunctionReturn(0);
  ierr = MatDestroy(&((*basis)->data.Theta));CHKERRQ(ierr);
  ierr = PetscFree((*basis)->data.names_data);CHKERRQ(ierr);
  ierr = PetscFree((*basis)->data.names);CHKERRQ(ierr);
  ierr = PetscFree(*basis);CHKERRQ(ierr);
  *basis = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyCreateBasisAndData(Vec x, PetscInt dim, PetscInt poly_order, PetscInt sine_order, Basis* basis)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SINDyCreateBasis(poly_order, sine_order, basis);CHKERRQ(ierr);
  ierr = SINDyCreateBasisData(*basis, x, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyInitializeBasesNames(Basis basis) {
  PetscErrorCode   ierr;
  PetscInt         b, r, o, d, names_size;
  PetscInt         *names_offsets;

  PetscFunctionBegin;
  if (basis->data.names) PetscFunctionReturn(0);

  /* Count string size. */
  ierr = PetscMalloc1(basis->data.B, &basis->data.names);CHKERRQ(ierr);
  ierr = PetscMalloc1(basis->data.B, &names_offsets);CHKERRQ(ierr);
  names_size = 0;
  b = 0;
  for (d = 0; d < basis->data.dim; d++) {    /* For each degree of freedom d. */
    /* Add basis functions using this degree of freedom. */
    for (o = 0; o <= basis->poly_order; o++) {
      names_offsets[b] = names_size;
      b++;
      names_size += 2 + PetscCeilReal(PetscLog10Real(o+2));
      /* Null character */
      names_size += 1;
    }
    for (o = 1; o <= basis->sine_order; o++) {
      names_offsets[b] = names_size;
      b++;
      names_size += 4 + PetscCeilReal(PetscLog10Real(o+2)) + 3;
      /* Null character */
      names_size += 1;
      names_offsets[b] = names_size;
      b++;
      names_size += 4 + PetscCeilReal(PetscLog10Real(o+2)) + 3;
      /* Null character */
      names_size += 1;
    }
  }

  /* Build string. */
  ierr = PetscMalloc1(names_size, &basis->data.names_data);CHKERRQ(ierr);
  b = 0;
  for (d = 0; d < basis->data.dim; d++) {    /* For each degree of freedom d. */
    /* Add basis functions using this degree of freedom. */
    for (o = 0; o <= basis->poly_order; o++) {
      basis->data.names[b] = basis->data.names_data + names_offsets[b];
      if (sprintf(basis->data.names[b], "%c^%d", 'a'+d, o) < 0) {
        PetscFunctionReturn(1);
      }
      b++;
    }
    for (o = 1; o <= basis->sine_order; o++) {
      basis->data.names[b] = basis->data.names_data + names_offsets[b];
      if (sprintf(basis->data.names[b], "sin(%d*%c)", o, 'a'+d) < 0) {
        PetscFunctionReturn(1);
      }
      b++;
      basis->data.names[b] = basis->data.names_data + names_offsets[b];
      if (sprintf(basis->data.names[b], "cos(%d*%c)", o, 'a'+d) < 0) {
        PetscFunctionReturn(1);
      }
      b++;
    }
  }
  ierr = PetscFree(names_offsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyCreateBasisData(Basis basis, Vec x, PetscInt dim)
{
  PetscErrorCode   ierr;
  PetscInt         x_size;
  const PetscReal* x_data;
  PetscInt         n, i, r, o, d;
  PetscReal        *Theta_data;
  PetscInt         *idn, *idb;

  /* Get data dimensions. */
  PetscFunctionBegin;
  ierr = VecGetSize(x, &x_size);CHKERRQ(ierr);
  basis->data.dim = dim;
  basis->data.N = x_size / dim;
  basis->data.B = SINDyCountBases(basis->data.dim, basis->poly_order, basis->sine_order);

  /* Compute basis data. */
  ierr = PetscMalloc1(basis->data.B * basis->data.N, &Theta_data);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &x_data);CHKERRQ(ierr);
  i = 0;
  for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
    r = n * dim;                   /* r is index into x_data for this data point. */
    for (d = 0; d < dim; d++) {    /* For each degree of freedom d. */
      /* Add basis functions using this degree of freedom. */
      for (o = 0; o <= basis->poly_order; o++) {
        Theta_data[i] = PetscPowRealInt(x_data[r+d], o);
        i++;
      }
      for (o = 1; o <= basis->sine_order; o++) {
        Theta_data[i] = PetscSinReal(o * x_data[r+d]);
        i++;
        Theta_data[i] = PetscCosReal(o * x_data[r+d]);
        i++;
      }
    }
  }
  ierr = VecRestoreArrayRead(x, &x_data);CHKERRQ(ierr);

  ierr = PetscMalloc2(x_size, &idn, basis->data.B, &idb);CHKERRQ(ierr);
  for (i=0;i<basis->data.N;i++) idn[i] = i;
  for (i=0;i<basis->data.B;i++) idb[i] = i;

  ierr = MatCreateSeqDense(PETSC_COMM_SELF, basis->data.N, basis->data.B, NULL, &basis->data.Theta);CHKERRQ(ierr);

  ierr = MatSetValues(basis->data.Theta,basis->data.N,idn,basis->data.B,idb,Theta_data,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(basis->data.Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(basis->data.Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(idn, idb);CHKERRQ(ierr);
  ierr = PetscFree(Theta_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDyFindSparseCoefficients(Basis basis, PetscInt dim, Vec* dxdt, Vec* Xis)
{
  PetscErrorCode ierr;
  PetscInt       d;

  PetscFunctionBegin;
  if (dim != basis->data.dim) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,"the given dim (=%d) must match the basis dim (=%d)", dim, basis->data.dim);
  }

  /* Run sparse least squares on each dimension of the data. */
  for (d = 0; d < dim; d++) {
    ierr = SINDySparseLeastSquares(basis->data.Theta, dxdt[d], NULL, Xis[d]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyBasisPrint(Basis basis, PetscInt dim, Vec* Xis)
{
  PetscErrorCode   ierr;
  const PetscReal  **xi_data;
  PetscInt         d, b;

  PetscFunctionBegin;
  if (dim != basis->data.dim) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,"the given dim (=%d) must match the basis dim (=%d)", dim, basis->data.dim);
  }

  ierr = SINDyInitializeBasesNames(basis);CHKERRQ(ierr);

  /* Get Xi data. */
  ierr = PetscMalloc1(dim, &xi_data);CHKERRQ(ierr);
  for (d = 0; d < dim; d++) {
    ierr = VecGetArrayRead(Xis[d], &xi_data[d]);CHKERRQ(ierr);
  }

  /* Print header line. */

  printf("%8s", "");
  for (d = 0; d < dim; d++) {
    printf("   %7s%c/dt", "d", 'a'+d);
  }
  printf("\n");

  /* Print results. */
  for (b = 0; b < basis->data.B; b++) {
    printf("%8s ", basis->data.names[b]);
    for (d = 0; d < dim; d++) {
      if (PetscAbsReal(xi_data[d][b]) < 5e-5) {
        printf("   %11s", "0");
      } else {
        printf("   % -9.4e", xi_data[d][b]);
      }
    }
    printf("\n");
  }

  /* Restore Xi data. */
  for (d = 0; d < dim; d++) {
    ierr = VecRestoreArrayRead(Xis[d], &xi_data[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(xi_data);CHKERRQ(ierr);
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
