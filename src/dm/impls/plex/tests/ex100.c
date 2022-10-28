static const char help[] = "Shows the bug in migrating the coordinate DS to the extruded coordinate DS";

#include <petscdmplex.h>
#include <petscsf.h>

#include <petsc/private/dmpleximpl.h>
/*
  run with 

  ./ex40 -dm_plex_shape box -dm_plex_dim 1 -dm_plex_box_faces 1,1,1 -post_label_dm_extrude 1

  to see the bug. 
*/
static PetscErrorCode LabelPoints(DM dm)
{
  DMLabel   label;
  PetscInt  pStart, pEnd, p;
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-label_mesh", &flg, NULL));
  if (!flg) PetscFunctionReturn(0);
  PetscCall(DMCreateLabel(dm, "test"));
  PetscCall(DMGetLabel(dm, "test", &label));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) PetscCall(DMLabelSetValue(label, p, p));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscDS ds; 
  DM      cdm; 

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMGetCoordinateDM(*dm,&cdm));
  PetscCall(DMGetDS(cdm,&ds)); 
  PetscCall(PetscDSView(ds,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(LabelPoints(*dm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "post_label_"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMGetCoordinateDM(*dm,&cdm));
  PetscCall(DMGetDS(cdm,&ds)); 
  PetscCall(PetscDSView(ds,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, NULL));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
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