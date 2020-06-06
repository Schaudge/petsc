static char help[] = "Tests correct layout and parallelization of DMPCreateGlobal/LocalCellVector_Plex.\n";
static char FILENAME[] = "ex35.c";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;
  PetscBool simplex, global;
  char      filename[PETSC_MAX_PATH_LEN]; /* The optional mesh file */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim         = 2;
  options->simplex     = PETSC_FALSE;
  options->global      = PETSC_TRUE;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, NULL, "DMCreateGlobal/LocalCellVector_Plex Options", "DM");CHKERRQ(ierr);
  {
    ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", FILENAME, options->dim, &options->dim, NULL, 1, 3);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-simplex", "Use simplex cells", "DMPlex", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-global_vector", "Retrieve global or local vector", "DM", options->global, &options->global, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-filename", "The mesh file", FILENAME, options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const char    *filename = user->filename;
  PetscInt       dim      = user->dim;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);}
  else      {ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);}
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test(DM dm, AppCtx *user)
{
  Vec               cVec;
  PetscInt          i, cHeight, cStart, cEnd;
  PetscInt          *ix;
  PetscScalar       *ival;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetVTKCellHeight(dm, &cHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc2(cEnd-cStart, &ix, cEnd-cStart, &ival);CHKERRQ(ierr);
  for (i = 0; i < cEnd-cStart; i++) {ix[i] = i;}
  if (user->global) {
    const PetscInt *idx;
    IS             glob;

    ierr = DMCreateGlobalCellVector(dm, &cVec);CHKERRQ(ierr);
    ierr = DMPlexGetCellNumbering(dm, &glob);CHKERRQ(ierr);
    ierr = ISGetIndices(glob, &idx);CHKERRQ(ierr);
    for (i = 0; i < cEnd-cStart; i++) {ival[i] = (PetscScalar) idx[i];}
    ierr = ISRestoreIndices(glob, &idx);CHKERRQ(ierr);
  } else {
    ierr = DMCreateLocalCellVector(dm, &cVec);CHKERRQ(ierr);
    for (i = 0; i < cEnd-cStart; i++) {ival[i] = (PetscScalar) i;}
  }
  ierr = VecSetValuesLocal(cVec, cEnd-cStart, (const PetscInt *) ix, (const PetscScalar *) ival, ADD_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(cVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(cVec);CHKERRQ(ierr);
  ierr = VecView(cVec, PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject) cVec)));CHKERRQ(ierr);
  ierr = PetscFree2(ix, ival);CHKERRQ(ierr);
  ierr = VecDestroy(&cVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  AppCtx         options;
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &options, &dm);CHKERRQ(ierr);
  ierr = test(dm, &options);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}

/*TEST
  testset:
    suffix: simplex_generated
    args: -orig_dm_view -dm_view -simplex
    args: -dm_plex_box_faces 5,5,5
    nsize: {{1 2 3}separate output}

    test:
      suffix: local
      args: -dim {{1 2 3}separate output} -global_vector false
      filter: sort -bnr

    test:
      suffix: global
      args: -dim {{1 2 3}separate output} -global_vector true

  testset:
    suffix: tensor_generated
    args: -orig_dm_view -dm_view -simplex false
    args: -dm_plex_box_faces 5,5,5
    nsize: {{1 2 3}separate output}

    test:
      suffix: local
      args:  -dim {{1 2 3}separate output} -global_vector false
      filter: sort -bnr

    test:
      suffix: global
      args: -dim {{1 2 3}separate output} -global_vector true
TEST*/
