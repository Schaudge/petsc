const char help[] = "Coverage and edge case test for LMVM";

#include <petscksp.h>
#include <petscmath.h>

int main(int argc, char **argv)
{
  PetscInt      type = 0, n = 10;
  Vec           x, g;
  Mat           B;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, "KSP");
  /* LMVM Types. 0: LMVMCDBFGS, 1: LMVMCDDFP, 2: LMVMCDQN */
  PetscCall(PetscOptionsInt("-type", "LMVM Type", __FILE__, type, &type, NULL));
  PetscOptionsEnd();
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &g));
  if (type == 0) {
    PetscCall(MatCreateLMVMCDBFGS(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &B));
  } else if (type == 1) {
    PetscCall(MatCreateLMVMCDDFP(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &B));
  } else if (type == 2) {
    PetscCall(MatCreateLMVMCDQN(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &B));
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Incompatible LMVM Type.");
  }
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLMVMAllocate(B, x, g));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&g));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

  test:
    suffix: 0
    output_file: output/lmvm_test.out
    args: -mat_lmvm_scale_type {{none scalar diagonal}} -type {{0 1 2}}

TEST*/
