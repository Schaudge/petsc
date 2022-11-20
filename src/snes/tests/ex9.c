static char help[] = "Create periodic mesh connected by hybrid cell.\n\n\n";

#include <petscdmplex.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  DM      hdm;
  DMLabel cutLabel;

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "orig_"));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Mesh"));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  PetscCall(DMGetLabel(*dm, "periodic_cut", &cutLabel));
  PetscCall(DMPlexCreateHybridMesh(*dm, cutLabel, NULL, 0, NULL, NULL, NULL, &hdm));
  PetscCall(DMDestroy(dm));
  PetscCall(PetscObjectSetName((PetscObject)hdm, "Hybrid Mesh"));
  PetscCall(DMSetFromOptions(hdm));
  PetscCall(DMViewFromOptions(hdm, NULL, "-dm_view"));
  *dm = hdm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -orig_dm_plex_simplex 0 -orig_dm_plex_box_bd periodic,none -orig_dm_plex_box_faces 5,1 \
            -orig_dm_plex_periodic_cut -orig_dm_view \
          -dm_view

TEST*/
