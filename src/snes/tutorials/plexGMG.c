static char help[] = "Poisson Problem d finite elements.\n\
We solve the Poisson problem discretized with finite elements\n\
using a hierarchy of parallel unstructured meshes (DMPLEX).\n\
This example supports automatic convergence estimation.\n\n\n";

/*
  This example can be run using the test system:

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*"
*/

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p1_gmg_vcycle
    requires:
    args:

TEST*/
