
#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

typedef struct {
  Vec diag;
} Mat_VectorDiagonal;

static PetscErrorCode MatAXPY_VectorDiagonal(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_VectorDiagonal *yctx = (Mat_VectorDiagonal *)Y->data;
  Mat_VectorDiagonal *xctx = (Mat_VectorDiagonal *)X->data;

  PetscFunctionBegin;
  PetscCall(VecAXPY(yctx->diag, a, xctx->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatEqual_VectorDiagonal(Mat Y, Mat X, PetscBool *equal)
{
  Mat_VectorDiagonal *yctx = (Mat_VectorDiagonal *)Y->data;
  Mat_VectorDiagonal *xctx = (Mat_VectorDiagonal *)X->data;

  PetscFunctionBegin;
  PetscCall(VecEqual(yctx->diag, xctx->diag, equal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRow_VectorDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)A->data;

  PetscFunctionBegin;
  if (ncols) *ncols = 1;
  if (cols) {
    PetscCall(PetscMalloc1(1, cols));
    (*cols)[0] = row;
  }
  if (vals) {
    PetscCall(PetscMalloc1(1, vals));
    PetscCall(VecGetValues(ctx->diag, 1, &row, *vals));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreRow_VectorDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  PetscFunctionBegin;
  if (ncols) *ncols = 0;
  if (cols) PetscCall(PetscFree(*cols));
  if (vals) PetscCall(PetscFree(*vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_VectorDiagonal(Mat A, Vec x, Vec y)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(VecPointwiseMult(y, ctx->diag, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_VectorDiagonal(Mat mat, Vec v1, Vec v2, Vec v3)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)mat->data;

  PetscFunctionBegin;
  if (v2 != v3) {
    PetscCall(VecPointwiseMult(v3, ctx->diag, v1));
    PetscCall(VecAXPY(v3, 1.0, v2));
  } else {
    Vec w;
    PetscCall(VecDuplicate(v2, &w));
    PetscCall(VecPointwiseMult(w, ctx->diag, v1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNorm_VectorDiagonal(Mat A, NormType type, PetscReal *nrm)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)A->data;

  PetscFunctionBegin;
  type = (type == NORM_FROBENIUS) ? NORM_2 : type;
  PetscCall(VecNorm(ctx->diag, type, nrm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_VectorDiagonal(Mat A, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])

{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatCreateSubMatrices(B, n, irow, icol, scall, submat));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_VectorDiagonal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_VectorDiagonal *actx = (Mat_VectorDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(*B, A, A));
  PetscCall(MatSetType(*B, MATVECTORDIAGONAL));
  PetscCall(PetscLayoutReference(A->rmap, &(*B)->rmap));
  PetscCall(PetscLayoutReference(A->cmap, &(*B)->cmap));
  if (op == MAT_COPY_VALUES) {
    Mat_VectorDiagonal *bctx = (Mat_VectorDiagonal *)(*B)->data;
    PetscCall(VecCopy(actx->diag, bctx->diag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMissingDiagonal_VectorDiagonal(Mat mat, PetscBool *missing, PetscInt *dd)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_VectorDiagonal(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mat->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_VectorDiagonal(Mat J, PetscViewer viewer)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)J->data;
  PetscBool             iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(VecView(ctx->diag, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyBegin_VectorDiagonal(Mat J, MatAssemblyType mt)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)J->data;

  PetscFunctionBegin;
  PetscCall(VecAssemblyBegin(ctx->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_VectorDiagonal(Mat J, MatAssemblyType mt)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)J->data;

  PetscFunctionBegin;
  PetscCall(VecAssemblyEnd(ctx->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_VectorDiagonal(Mat J, Vec x)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)J->data;

  PetscFunctionBegin;
  PetscCall(VecCopy(ctx->diag, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalScale_VectorDiagonal(Mat J, Vec L, Vec R)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)J->data;

  PetscFunctionBegin;
  if (L) {
    PetscCall(VecPointwiseMult(ctx->diag, L, ctx->diag));
  }
  if (R) {
    PetscCall(VecPointwiseMult(ctx->diag, R, ctx->diag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode MatDiagonalSet_VectorDiagonal(Mat J, Vec D, InsertMode is)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)J->data;

  PetscFunctionBegin;
  switch (is) {
  case ADD_VALUES:
  case ADD_ALL_VALUES:
  case ADD_BC_VALUES:
    PetscCall(VecAXPY(ctx->diag, 1.0, D));
    break;
  case INSERT_VALUES:
  case INSERT_BC_VALUES:
  case INSERT_ALL_VALUES:
    PetscCall(VecCopy(D, ctx->diag));
    break;
  case MAX_VALUES:
    PetscCall(VecPointwiseMax(ctx->diag, D, ctx->diag));
    break;
  case MIN_VALUES:
    PetscCall(VecPointwiseMin(ctx->diag, D, ctx->diag));
    break;
  case NOT_SET_VALUES:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatShift_VectorDiagonal(Mat Y, PetscScalar a)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)Y->data;

  PetscFunctionBegin;
  PetscCall(VecShift(ctx->diag, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_VectorDiagonal(Mat Y, PetscScalar a)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)Y->data;

  PetscFunctionBegin;
  PetscCall(VecScale(ctx->diag, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_VectorDiagonal(Mat Y)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)Y->data;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(ctx->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_VectorDiagonal(Mat matin, Vec b, Vec x)
{
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *)matin->data;

  PetscFunctionBegin;
  PetscCall(VecPointwiseDivide(x, b, ctx->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetInfo_VectorDiagonal(Mat A, MatInfoType flag, MatInfo *info)
{
  PetscFunctionBegin;
  info->block_size   = 1.0;
  info->nz_allocated = 1.0;
  info->nz_used      = 1.0;
  info->nz_unneeded  = 0.0;
  info->assemblies   = A->num_ass;
  info->mallocs      = 0.0;
  info->memory       = 0; /* REVIEW ME */
  if (A->factortype) {
    info->fill_ratio_given  = 1.0;
    info->fill_ratio_needed = 1.0;
    info->factor_mallocs    = 0.0;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCreateVectorDiagonal - Creates a matrix with given vector on its diagonal.

   Collective

   Input Parameters:
.  diag - Vector for the diagonal

   Output Parameter:
.  J - the diagonal matrix

   Level: advanced

   Notes:
    Only supports square matrices with the same number of local rows and columns

.seealso: [](chapter_matrices), `Mat`, `MatDestroy()`, `MATCONSTANTDIAGONAL`, `MatScale()`, `MatShift()`, `MatMult()`, `MatGetDiagonal()`, `MatGetFactor()`, `MatSolve()`
@*/
PetscErrorCode MatCreateVectorDiagonal(Vec diag, Mat *J)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(diag, VEC_CLASSID, 1);
  PetscCall(MatCreate(PetscObjectComm((PetscObject) diag), J));
  PetscInt m, M;
  PetscCall(VecGetLocalSize(diag, &m));
  PetscCall(VecGetSize(diag, &M));
  PetscCall(MatSetSizes(*J, m, m, M, M));
  PetscCall(MatSetType(*J, MATVECTORDIAGONAL));
  PetscCall(PetscLayoutReference(diag->map, &(*J)->cmap));
  PetscCall(PetscLayoutReference(diag->map, &(*J)->rmap));
  Mat_VectorDiagonal *ctx = (Mat_VectorDiagonal *) (*J)->data;
  PetscCall(PetscObjectReference((PetscObject)diag));
  PetscCall(VecDestroy(&ctx->diag));
  ctx->diag = diag;
  VecType type;
  PetscCall(VecGetType(diag, &type));
  PetscCall(PetscStrallocpy(type, &(*J)->defaultvectype));
  PetscCall(MatSetUp(*J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_VectorDiagonal(Mat A)
{
  Mat_VectorDiagonal *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  PetscInt m, M;
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatGetLocalSize(A, &M, NULL));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)A), &ctx->diag));
  PetscCall(VecSetSizes(ctx->diag, m, M));
  PetscCall(PetscLayoutReference(A->rmap, &ctx->diag->map));

  A->data   = (void *)ctx;

  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_TRUE;
  A->structurally_symmetric = PETSC_BOOL3_TRUE;
  A->structural_symmetry_eternal = PETSC_TRUE;
  A->symmetry_eternal = PETSC_TRUE;

  A->ops->mult              = MatMult_VectorDiagonal;
  A->ops->multadd           = MatMultAdd_VectorDiagonal;
  A->ops->multtranspose     = MatMult_VectorDiagonal;
  A->ops->multtransposeadd  = MatMultAdd_VectorDiagonal;
  A->ops->norm              = MatNorm_VectorDiagonal;
  A->ops->createsubmatrices = MatCreateSubMatrices_VectorDiagonal;
  A->ops->duplicate         = MatDuplicate_VectorDiagonal;
  A->ops->missingdiagonal   = MatMissingDiagonal_VectorDiagonal;
  A->ops->getrow            = MatGetRow_VectorDiagonal;
  A->ops->restorerow        = MatRestoreRow_VectorDiagonal;
  A->ops->solve             = MatSolve_VectorDiagonal;
  A->ops->solvetranspose    = MatSolve_VectorDiagonal;
  A->ops->shift             = MatShift_VectorDiagonal;
  A->ops->scale             = MatScale_VectorDiagonal;
  A->ops->getdiagonal       = MatGetDiagonal_VectorDiagonal;
  A->ops->diagonalset       = MatDiagonalSet_VectorDiagonal;
  A->ops->diagonalscale     = MatDiagonalScale_VectorDiagonal;
  A->ops->view              = MatView_VectorDiagonal;
  A->ops->zeroentries       = MatZeroEntries_VectorDiagonal;
  A->ops->assemblybegin     = MatAssemblyBegin_VectorDiagonal;
  A->ops->assemblyend       = MatAssemblyEnd_VectorDiagonal;
  A->ops->destroy           = MatDestroy_VectorDiagonal;
  A->ops->getinfo           = MatGetInfo_VectorDiagonal;
  A->ops->equal             = MatEqual_VectorDiagonal;
  A->ops->axpy              = MatAXPY_VectorDiagonal;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATVECTORDIAGONAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

