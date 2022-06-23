const char help[] = "Visualize PetscFE finite elements with plotly.js";

#include <petsc.h>
#include <mongoose.h>

static PetscErrorCode PetscFEView_Mongoose(PetscFE fe)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt  dim = 3;
  PetscInt  Nc = 1;
  PetscBool is_simplex = PETSC_TRUE;
  PetscInt  qorder = PETSC_DETERMINE;
  PetscFE   fe;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscFECreateDefault(PETSC_COMM_WORLD, dim, Nc, is_simplex, NULL, qorder, &fe));
  PetscCall(PetscFEView_Mongoose(fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
