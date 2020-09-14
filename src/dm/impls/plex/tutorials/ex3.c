#include <petscdmplex.h>

typedef struct {
  PetscBool interp, distribute, simplex;
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
  PetscInt  dim;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->interp      = PETSC_TRUE;
  options->distribute  = PETSC_TRUE;
  options->simplex     = PETSC_TRUE;
  options->filename[0] = '\0';
  options->dim         = 2;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex2.c", options->interp, &options->interp, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Generate intermediate mesh elements", "ex2.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Generate intermediate mesh elements", "ex2.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex2.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The dimension of problem used for non-file mesh", "ex2.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *opts, DM *dm)
{
  const PetscInt numComp[1] = {1}, bcField[1] = {0};
  const PetscInt numFields = 1, numBC = 1;
  IS             bcPointIS[1];
  PetscInt       *numDof;
  PetscInt       dim;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (opts->filename[0]) {
    ierr = DMPlexCreateFromFile(comm, opts->filename, opts->interp, dm);CHKERRQ(ierr);
  } else {
    DMLabel label;

    ierr = DMPlexCreateBoxMesh(comm, opts->dim, opts->simplex, NULL, NULL, NULL, NULL, opts->interp, dm);CHKERRQ(ierr);
    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
  }

  ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  ierr = PetscCalloc1(numFields*(dim+1), &numDof);CHKERRQ(ierr);
  /* Let u be defined on vertices */
  numDof[0*(dim+1)+0]     = 1;
  /* Prescribe a Dirichlet condition on u on the boundary
   Label "marker" is made by the mesh creation routine */
  ierr = DMGetStratumIS(*dm, "marker", 1, &bcPointIS[0]);CHKERRQ(ierr);
  ierr = DMSetNumFields(*dm, numFields);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(*dm, NULL, numComp, numDof, numBC, bcField, NULL, bcPointIS, NULL, &section);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPointIS[0]);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 0, "u");CHKERRQ(ierr);
  /* Tell the DM to use this data layout */
  ierr = DMSetLocalSection(*dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  if (opts->distribute) {
    DM dmdist;

    ierr = DMPlexDistribute(*dm, 0, NULL, &dmdist);CHKERRQ(ierr);
    if (dmdist) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = dmdist;
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx          opts;
  DM              dm;
  DM              *dmreg;
  PetscInt        n;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &opts);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &opts, &dm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "---------- DM VIEW BEFORE ----------\n");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMPlexConstructCohesiveRegions(dm, NULL, &n, &dmreg);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "---------- DM VIEW AFTER  ----------\n");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
