#include "contexts.cxx"
#include <petscmat.h>
#include <adolc/adolc_sparse.h>

/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/

/*
  Basic printing for sparsity pattern

  Input parameters:
  comm     - MPI communicator
  m        - number of rows

  Output parameter:
  sparsity - matrix sparsity pattern, typically computed using an ADOL-C function such as jac_pat
*/
PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **sparsity)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"Sparsity pattern:\n");CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %2d: ",i);CHKERRQ(ierr);
    for(j=1; j<= (PetscInt) sparsity[i][0] ;j++)
      ierr = PetscPrintf(comm," %2d ",sparsity[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Generate a seed matrix defining the partition of columns of a matrix by a particular coloring,
  used for matrix compression

  Input parameter:
  iscoloring - the index set coloring to be used

  Output parameter:
  S          - the resulting seed matrix

  Notes:
  Before calling GenerateSeedMatrix, Seed should be allocated as a logically 2d array with number of
  rows equal to the matrix to be compressed and number of columns equal to the number of colors used
  in iscoloring.
*/
PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscScalar **S)
{
  PetscErrorCode ierr;
  IS             *is;
  PetscInt       p,size,colour,j;
  const PetscInt *indices;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&p,&is);CHKERRQ(ierr);
  for (colour=0; colour<p; colour++) {
    ierr = ISGetLocalSize(is[colour],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
    for (j=0; j<size; j++)
      S[indices[j]][colour] = 1;
    ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Establish a look-up vector whose entries contain the colour used for that diagonal entry. Clearly
  we require the number of dependent and independent variables to be equal in this case.
  Input parameters:
  S        - the seed matrix defining the coloring
  sparsity - the sparsity pattern of the matrix to be recovered, typically computed using an ADOL-C
             function, such as jac_pat or hess_pat
  m        - the number of rows of Seed (and the matrix to be recovered)
  p        - the number of colors used (also the number of columns in Seed)
  Output parameter:
  R        - the recovery vector to be used for de-compression
*/
PetscErrorCode GenerateSeedMatrixPlusRecovery(ISColoring iscoloring,PetscScalar **S,PetscInt *R)
{
  PetscErrorCode ierr;
  IS             *is;
  PetscInt       p,size,colour,j;
  const PetscInt *indices;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&p,&is);CHKERRQ(ierr);
  for (colour=0; colour<p; colour++) {
    ierr = ISGetLocalSize(is[colour],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
    for (j=0; j<size; j++) {
      S[indices[j]][colour] = 1.;
      R[indices[j]] = colour;
    }
    ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Establish a look-up matrix whose entries contain the column coordinates of the corresponding entry
  in a matrix which has been compressed using the coloring defined by some seed matrix

  Input parameters:
  sparsity  - the sparsity pattern of the matrix to be recovered, typically computed using an ADOL-C
              function, such as jac_pat or hess_pat
  ctx       - AdolcCtx object containing number of rows, number of colours, seed matrix and CSR
              vectors

  Output parameters:
  adctx->ri - the row index component of the CSR recovery matrix to be used for de-compression
  adctx->rj - the column index component of the CSR recovery matrix to be used for de-compression
  adctx->r  - the values of the CSR recovery matrix to be used for de-compression
*/
PetscErrorCode GetRecoveryMatrix(unsigned int **sparsity,void *ctx)
{
  AdolcCtx *adctx = (AdolcCtx*)ctx;
  PetscInt i,j,jj = 0,k,nnz,colour,m = adctx->m,p = adctx->p;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      nnz = (PetscInt) sparsity[i][0];        // Number of nonzeros on row i
      if (colour == 0) adctx->ri[0][i] = jj;  // Store counter for first nonzero on compressed row i
      for (k=1; k<=nnz; k++) {
        j = (PetscInt) sparsity[i][k];        // Column index of k^th nonzero on row i
        if (adctx->Seed[j][colour] == 1.) {
          adctx->r[0][jj] = j;                // Store index of k^th nonzero on row i
          adctx->rj[0][jj] = colour;          // Store colour of k^th nonzero on row i
          jj++;                               // Counter for nonzeros
          break;
        }
      }
    }
  }
  adctx->ri[0][m] = jj;  // NOTE: This is number of nonzeros in matrix itself
  PetscFunctionReturn(0);
}

/*
  Convert an m x p compressed Jacobian into CSR format, using the CSR form recovery matrix defined by
  integer vectors Ri and Rj.

  NOTE:
   CSR decomposition of recovery matrix is given by Ri, Rj, R
   CSR decomposition of compressed Jac is given by Ri, R, c

  Input parameters:
  C   - compressed matrix to recover values from
  a   - shift value for implicit problems (select NULL or unity for explicit problems)
  ctx - AdolcCtx containing number of rows and CSR vectors

  Output parameter:
  c   - CSR vector version of compressed Jacobian
*/
PetscErrorCode ConvertToCSR(PetscScalar **C,PetscScalar **c,PetscReal *a,void *ctx)
{
  AdolcCtx *adctx = (AdolcCtx*)ctx;
  PetscInt i,j;

  PetscFunctionBegin;
  for (i=0; i<adctx->m; i++) {
    for (j=adctx->ri[0][i]; j<adctx->ri[0][i+1]; j++) {
      c[0][j] = C[i][adctx->rj[0][j]];  // rj[0][j] = colour of j^th nonzero on i^th compressed row
      if (a) c[0][j] *= *a;
    }
  }
  PetscFunctionReturn(0);
}

/*
  Recover the values of a sparse matrix from a compressed format and insert these into a matrix

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  c    - CSR vector version of compressed Jacobian
  ctx  - AdolcCtx containing number of rows and CSR vectors

  Output parameter:
  A    - Mat to be populated with values from compressed matrix
*/
PetscErrorCode RecoverJacobian(Mat A,InsertMode mode,PetscScalar **c,void *ctx)
{
  PetscErrorCode ierr;
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j;

  PetscFunctionBegin;
  for (i=0; i<adctx->m; i++) {
    for (j=adctx->ri[0][i]; j<adctx->ri[0][i+1]; j++) {
      ierr = MatSetValues(A,1,&i,1,&adctx->r[0][j],&c[0][j],mode);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  Recover the values of the local portion of a sparse matrix from a compressed format and insert
  these into the local portion of a matrix

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  c    - CSR vector version of compressed Jacobian
  ctx  - AdolcCtx containing number of rows and CSR vectors

  Output parameter:
  A    - Mat to be populated with values from compressed matrix
*/
PetscErrorCode RecoverJacobianLocal(Mat A,InsertMode mode,PetscScalar **c,void *ctx)
{
  PetscErrorCode ierr;
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j;

  PetscFunctionBegin;
  for (i=0; i<adctx->m; i++) {
    for (j=adctx->ri[0][i]; j<adctx->ri[0][i+1]; j++) {
      ierr = MatSetValuesLocal(A,1,&i,1,&adctx->r[0][j],&c[0][j],mode);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

// TODO: split inputs so this will work
// TODO: test it works
// TODO: documentation
PetscErrorCode RecoverJacobianWithArrays(Mat A,PetscScalar **c,void *ctx)
{
  PetscErrorCode ierr;
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MatUpdateMPIAIJWithArrays(A,adctx->l,adctx->l,PETSC_DETERMINE,PETSC_DETERMINE,adctx->ri[rank],adctx->r[rank],c[rank]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Recover the diagonal of the Jacobian from its compressed matrix format

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  r    - recovery vector to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  diag - Vec to be populated with values from compressed matrix
*/
PetscErrorCode RecoverDiagonal(Vec diag,InsertMode mode,PetscInt m,PetscInt **r,PetscScalar **C,PetscReal *a)
{
  PetscErrorCode ierr;
  PetscInt       i,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    colour = (PetscInt)r[0][i];
    if (a)
      C[i][colour] *= *a;
    ierr = VecSetValues(diag,1,&i,&C[i][colour],mode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Recover the local portion of the diagonal of the Jacobian from its compressed matrix format

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  r    - recovery vector to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  diag - Vec to be populated with values from compressed matrix
*/
PetscErrorCode RecoverDiagonalLocal(Vec diag,InsertMode mode,PetscInt m,PetscInt **r,PetscScalar **C,PetscReal *a)
{
  PetscErrorCode ierr;
  PetscInt       i,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    colour = (PetscInt)r[0][i];
    if (a)
      C[i][colour] *= *a;
    ierr = VecSetValuesLocal(diag,1,&i,&C[i][colour],mode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
