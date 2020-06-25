#include <petscdmplex.h>

typedef struct {
  char      filename[PETSC_MAX_PATH_LEN];
  PetscBool interp, dist, simplex, isA;
  PetscInt  dim, overlap;
  PetscInt  faces[3];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, PetscBool isA, AppCtx *options)
{
  PetscErrorCode ierr;
  PetscInt       n = 3;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';
  options->isA         = isA;
  options->interp      = PETSC_TRUE;
  options->dist        = PETSC_TRUE;
  options->simplex     = PETSC_FALSE;
  options->overlap     = 0;
  options->dim         = 2;
  options->faces[0]    = 2;
  options->faces[1]    = 2;
  options->faces[2]    = 2;

  ierr = PetscOptionsBegin(comm, isA ? "a_" : "b_", "Mesh Combine Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex9.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate mesh", "ex9.c", options->interp, &options->interp, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute mesh", "ex9.c", options->dist, &options->dist, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Create simplex mesh", "ex9.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-overlap", "Number of overlap levels for dm", "ex9.c", options->overlap, &options->overlap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "Dimension of mesh", "ex9.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-mesh_box_faces", "Number of box mesh faces per dimension", "DMPlexCreateBoxMesh", options->faces, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, PetscBool isA, AppCtx *user, DM *dm)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (user->filename[0]) {
    ierr = DMPlexCreateFromFile(comm, user->filename, user->interp, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, user->faces, NULL, NULL, NULL, user->interp, dm);CHKERRQ(ierr);
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  if (user->dist) {
    DM  dmDist = NULL;
    ierr = DMPlexDistribute(*dm, user->overlap, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = dmDist;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, user->isA ? "MeshA" : "MeshB");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, user->isA ? "-a_dm_view" : "-b_dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  MPI_Comm        comm;
  AppCtx          usera, userb;
  DM              dma, dmb, dmc;
  DMLabel         lbla, lblb;
  IS              va, vb;
  const PetscInt  *aind, *bind;
  PetscInt        i, aissize, bissize;

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = ProcessOptions(comm, PETSC_TRUE, &usera);CHKERRQ(ierr);
  ierr = ProcessOptions(comm, PETSC_FALSE, &userb);CHKERRQ(ierr);
  ierr = CreateMesh(comm, PETSC_TRUE, &usera, &dma);CHKERRQ(ierr);
  ierr = CreateMesh(comm, PETSC_FALSE, &userb, &dmb);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "-a_label", &lbla);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "-b_label", &lblb);CHKERRQ(ierr);

  ierr = DMPlexGetVertexNumbering(dma, &va);CHKERRQ(ierr);
  ierr = DMPlexGetVertexNumbering(dmb, &vb);CHKERRQ(ierr);
  ierr = ISViewFromOptions(va, NULL, "-a_is_view");CHKERRQ(ierr);
  ierr = ISViewFromOptions(vb, NULL, "-b_is_view");CHKERRQ(ierr);
  ierr = ISGetIndices(va, &aind);CHKERRQ(ierr);
  ierr = ISGetIndices(vb, &bind);CHKERRQ(ierr);
  ierr = ISGetLocalSize(va, &aissize);CHKERRQ(ierr);
  ierr = ISGetLocalSize(vb, &bissize);CHKERRQ(ierr);
  for (i = 0; i < aissize; i++) {
    if (aind[i] == 0 || aind[i] == 1 || aind[i] == 2) {
      ierr = DMLabelSetValue(lbla, aind[i], 1);CHKERRQ(ierr);
    }
  }
  for (i = 0; i < bissize; i++) {
    if (bind[i] == 0 || bind[i] == 1 || bind[i] == 2) {
      ierr = DMLabelSetValue(lblb, bind[i], 1);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(va, &aind);CHKERRQ(ierr);
  ierr = ISRestoreIndices(vb, &bind);CHKERRQ(ierr);
  ierr = DMLabelView(lbla, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMLabelView(lblb, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMPlexCombine(dma, dmb, lbla, lblb, &dmc);CHKERRQ(ierr);

  ierr = DMViewFromOptions(dmc, NULL, "-dm_view_c");CHKERRQ(ierr);
  ierr = DMLabelDestroy(&lbla);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&lblb);CHKERRQ(ierr);
  ierr = DMDestroy(&dma);CHKERRQ(ierr);
  ierr = DMDestroy(&dmb);CHKERRQ(ierr);
  ierr = DMDestroy(&dmc);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
