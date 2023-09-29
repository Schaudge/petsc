const char help[] = "Profile the performance of MATLMVM MatSolve() in a loop";

#include <petscksp.h>

int main(int argc, char **argv)
{
  PetscInt      n        = 1000;
  PetscInt      n_epochs = 10;
  PetscInt      n_iters  = 10;
  Vec           x, g, dx, df, p;
  PetscRandom   rand;
  PetscLogStage matsolve_loop, main_stage;
  Mat           B;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, "KSP");
  PetscCall(PetscOptionsInt("-n", "Vector size", __FILE__, n, &n, NULL));
  PetscCall(PetscOptionsInt("-epochs", "Number of epochs", __FILE__, n_epochs, &n_epochs, NULL));
  PetscCall(PetscOptionsInt("-iters", "Number of iterations per epoch", __FILE__, n_iters, &n_iters, NULL));
  PetscOptionsEnd();
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecDuplicate(x, &dx));
  PetscCall(VecDuplicate(x, &df));
  PetscCall(VecDuplicate(x, &p));
  PetscCall(MatCreateLMVMBFGS(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &B));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLMVMAllocate(B, x, g));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscLogStageRegister("LMVM MatSolve Loop", &matsolve_loop));
  PetscCall(PetscLogStageGetId("Main Stage", &main_stage));
  PetscCall(PetscLogStageSetVisible(main_stage, PETSC_FALSE));
  for (PetscInt epoch = 0; epoch < n_epochs + 1; epoch++) {
    PetscScalar dot;
    PetscReal   xscale, fscale, absdot;

    PetscCall(VecSetRandom(dx, rand));
    PetscCall(VecSetRandom(df, rand));
    PetscCall(VecDot(dx, df, &dot));
    absdot = PetscAbsScalar(dot);
    PetscCall(VecZeroEntries(x));
    PetscCall(VecZeroEntries(g));
    xscale = 1.0;
    fscale = absdot / dot;

    if (epoch > 0) PetscCall(PetscLogStagePush(matsolve_loop));
    PetscCall(MatLMVMUpdate(B, x, g));
    for (PetscInt iter = 0; iter < n_iters; iter++, xscale *= -1.0, fscale *= -1.0) {

      PetscCall(VecAXPY(x, xscale, dx));
      PetscCall(VecAXPY(g, fscale, df));
      PetscCall(MatLMVMUpdate(B, x, g));
      PetscCall(MatSolve(B, g, p));
    }
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (epoch > 0) PetscCall(PetscLogStagePop());
  }
  PetscCall(MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&df));
  PetscCall(VecDestroy(&dx));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -mat_lmvm_scale_type none

TEST*/
