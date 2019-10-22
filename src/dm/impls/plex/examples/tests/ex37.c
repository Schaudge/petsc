static char help[] = "Tests for section orderings\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;         /* The topological mesh dimension */
  PetscBool cellSimplex; /* Use simplices or hexes */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim         = 2;
  options->cellSimplex = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex37.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex37.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->cellSimplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDiscretization(DM dm, AppCtx *user)
{
  PetscFE        feA, feB;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), user->dim, 1, user->cellSimplex, "a_", -1, &feA);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feA, "field A");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) feA);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), user->dim, user->dim, user->cellSimplex, "b_", -1, &feB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feB, "field B");CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) feB);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feA);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   s;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = CreateDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: p2_def
    args: -dm_plex_box_faces 1,1 -dm_view -a_petscspace_degree 2 -b_petscspace_degree 2 -petscds_view -section_view

  test:
    suffix: p2_fm
    args: -dm_plex_box_faces 1,1 -dm_view -a_petscspace_degree 2 -b_petscspace_degree 2 -petscds_view -petscsection_point_major 0 -section_view

  test:
    suffix: p2_cm
    args: -dm_plex_box_faces 1,1 -dm_view -a_petscspace_degree 2 -b_petscspace_degree 2 -petscds_view -petscsection_chunk_major -section_view

  test:
    suffix: p3_def
    args: -dm_plex_box_faces 1,1 -dm_view -a_petscspace_degree 3 -b_petscspace_degree 3 -petscds_view -section_view

  test:
    suffix: p3_fm
    args: -dm_plex_box_faces 1,1 -dm_view -a_petscspace_degree 3 -b_petscspace_degree 3 -petscds_view -petscsection_point_major 0 -section_view

  test:
    suffix: p3_cm
    args: -dm_plex_box_faces 1,1 -dm_view -a_petscspace_degree 3 -b_petscspace_degree 3 -petscds_view -petscsection_chunk_major -section_view

TEST*/
