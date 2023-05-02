const char help[] = "Test overhead of PETSc abstractions";

#include <petsctao.h>

static PetscErrorCode ObjAndGrad(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscFunctionBegin;
  *f = 0.0;
  PetscCall(VecNorm(x, NORM_2, f));
  *f = 0.5 * (*f) * (*f);
  PetscCall(VecCopy(x, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NoOp(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt n_iter = 1;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, NULL);
  PetscCall(PetscOptionsInt("-n_iter", "Number of iterations in each stage", NULL, n_iter, &n_iter, NULL));
  PetscOptionsEnd();
  Mat eye;
  PetscCall(MatCreateConstantDiagonal(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 1.0, &eye));
  Vec sol;
  PetscCall(MatCreateVecs(eye, &sol, NULL));
  Tao tao;
  PetscCall(VecSet(sol, 1.0));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOSNES));
  PetscCall(TaoSetSolution(tao, sol));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, ObjAndGrad, NULL));
  PetscCall(TaoSetHessian(tao, eye, eye, NoOp, NULL));
  SNES snes;
  PetscCall(TaoSNESGetSNES(tao, &snes));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  KSP ksp;
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMAT));
  PetscCall(PCMatSetApplyOperation(pc, MATOP_SOLVE));
  PetscCall(PCSetOperators(pc, eye, eye));
  PetscLogStage after_first_solve;
  PetscCall(PetscLogStageRegister("After first solve", &after_first_solve));
  for (PetscInt i = 0; i < n_iter; i++) {
    PetscCall(VecSet(sol,1.0));
    PetscCall(TaoSolve(tao));
    if (i == 0) PetscCall(PetscLogStagePush(after_first_solve));
  }
  PetscCall(PetscLogStagePop());
  PetscCall(VecDestroy(&sol));
  PetscCall(MatDestroy(&eye));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
