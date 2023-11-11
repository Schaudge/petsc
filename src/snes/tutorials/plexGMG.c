static char help[] = "Poisson Problem d finite elements.\n\
We solve the Poisson problem discretized with finite elements\n\
using a hierarchy of parallel unstructured meshes (DMPLEX).\n\
This example supports automatic convergence estimation.\n\n\n";

/*
  This example can be run using the test system:

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*"

  You can see the help using

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-help"

  The mesh can be viewed using

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-dm_view"

  and many formats are supported such as VTK, HDF5, and drawing,

    make -f ./gmakefile test search="snes_tutorials-plexGMG_*" EXTRA_OPTIONS="-dm_view draw -draw_pause -1"
*/

#include <petscdmplex.h>

typedef struct {
  PetscBool viewError; // Output the solution error
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->viewError = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-error_view", "Output the solution error", "plexGMG.c", options->viewError, &options->viewError, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;   // DMPLEX mesh
  AppCtx user; // User-defined work context

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p1_gmg_vcycle
    requires: triangle
    args:

TEST*/
