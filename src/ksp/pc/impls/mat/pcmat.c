
#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct _PCMAT {
  MatOperation solve;
} PC_Mat;

static PetscErrorCode PCApply_Mat(PC pc, Vec x, Vec y)
{
  PC_Mat *pcmat = (PC_Mat *) pc->data;

  PetscFunctionBegin;
  switch (pcmat->solve) {
  case MATOP_MULT:
    PetscCall(MatMult(pc->pmat, x, y));
    break;
  case MATOP_MULT_TRANSPOSE:
    PetscCall(MatMultTranspose(pc->pmat, x, y));
    break;
  case MATOP_SOLVE:
    PetscCall(MatSolve(pc->pmat, x, y));
    break;
  case MATOP_SOLVE_TRANSPOSE:
    PetscCall(MatSolveTranspose(pc->pmat, x, y));
    break;
  case MATOP_MULT_HERMITIAN_TRANSPOSE:
    PetscCall(MatMultHermitianTranspose(pc->pmat, x, y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Operation cannot be PCApply operation for PCMAT");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApply_Mat(PC pc, Mat X, Mat Y)
{
  PC_Mat *pcmat = (PC_Mat *) pc->data;

  PetscFunctionBegin;
  switch (pcmat->solve) {
  case MATOP_MULT:
    PetscCall(MatMatMult(pc->pmat, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    break;
  case MATOP_MULT_TRANSPOSE:
    PetscCall(MatTransposeMatMult(pc->pmat, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    break;
  case MATOP_SOLVE:
    PetscCall(MatMatSolve(pc->pmat, X, Y));
    break;
  case MATOP_SOLVE_TRANSPOSE:
    PetscCall(MatMatSolveTranspose(pc->pmat, X, Y));
    break;
  case MATOP_MULT_HERMITIAN_TRANSPOSE:
    {
      Mat W;

      PetscCall(MatDuplicate(X, MAT_COPY_VALUES, &W));
      PetscCall(MatConjugate(W));
      PetscCall(MatTransposeMatMult(pc->pmat, W, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
      PetscCall(MatConjugate(Y));
      PetscCall(MatDestroy(&W));
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Operation cannot be PCMatApply operation for PCMAT");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_Mat(PC pc, Vec x, Vec y)
{
  PC_Mat *pcmat = (PC_Mat *) pc->data;

  PetscFunctionBegin;
  switch (pcmat->solve) {
  case MATOP_MULT:
    PetscCall(MatMultTranspose(pc->pmat, x, y));
    break;
  case MATOP_MULT_TRANSPOSE:
    PetscCall(MatMult(pc->pmat, x, y));
    break;
  case MATOP_SOLVE:
    PetscCall(MatSolveTranspose(pc->pmat, x, y));
    break;
  case MATOP_SOLVE_TRANSPOSE:
    PetscCall(MatSolve(pc->pmat, x, y));
    break;
  case MATOP_MULT_HERMITIAN_TRANSPOSE:
    {
      Vec w;

      PetscCall(VecDuplicate(x, &w));
      PetscCall(VecCopy(x, w));
      PetscCall(VecConjugate(w));
      PetscCall(MatMult(pc->pmat, w, y));
      PetscCall(VecConjugate(y));
      PetscCall(VecDestroy(&w));
    }
    PetscCall(MatMultHermitianTranspose(pc->pmat, x, y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Operation cannot be PCApply operation for PCMAT");
  }
  PetscFunctionBegin;
  PetscCall(MatMultTranspose(pc->pmat, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Mat(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatSetSolveOperation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatGetSolveOperation_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCMatSetSolveOperation(PC pc, MatOperation matop)
{
  PetscFunctionBegin;
  PetscTryMethod((PetscObject)pc, "PCMatSetSolveOperation_C", (PC,MatOperation),(pc,matop));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCMatGetSolveOperation(PC pc, MatOperation *matop)
{
  PetscFunctionBegin;
  PetscTryMethod((PetscObject)pc, "PCMatGetSolveOperation_C", (PC,MatOperation*),(pc,matop));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatSetSolveOperation_Mat(PC pc, MatOperation matop)
{
  PC_Mat *pcmat = (PC_Mat *) pc->data;

  PetscFunctionBegin;
  switch (matop) {
  case MATOP_MULT:
  case MATOP_MULT_TRANSPOSE:
  case MATOP_SOLVE:
  case MATOP_SOLVE_TRANSPOSE:
  case MATOP_MULT_HERMITIAN_TRANSPOSE:
    pcmat->solve = matop;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Operation cannot be PCApply operation for PCMAT");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatGetSolveOperation_Mat(PC pc, MatOperation *matop)
{
  PC_Mat *pcmat = (PC_Mat *) pc->data;

  PetscFunctionBegin;
  *matop = pcmat->solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCMAT - A preconditioner obtained by multiplying by the preconditioner matrix supplied
             in `PCSetOperators()` or `KSPSetOperators()`

   Note:
    This one is a little strange. One rarely has an explicit matrix that approximates the
         inverse of the matrix they wish to solve for.

   Level: intermediate

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCSHELL`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Mat(PC pc)
{
  PetscFunctionBegin;
  PC_Mat * data;
  PetscCall(PetscNew(&data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatSetSolveOperation_C", PCMatSetSolveOperation_Mat));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatGetSolveOperation_C", PCMatGetSolveOperation_Mat));
  data->solve = MATOP_MULT;
  pc->data = data;
  pc->ops->apply               = PCApply_Mat;
  pc->ops->matapply            = PCMatApply_Mat;
  pc->ops->applytranspose      = PCApplyTranspose_Mat;
  pc->ops->setup               = NULL;
  pc->ops->destroy             = PCDestroy_Mat;
  pc->ops->setfromoptions      = NULL;
  pc->ops->view                = NULL;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
