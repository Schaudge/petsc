static const char help[] = "Test for parallel interpolation";

#include <petscdmplex.h>
#include <petscsf.h>

#include <petsc/private/dmpleximpl.h>

typedef struct {
  PetscInt  meshNum; /* The test mesh number */
  PetscBool orient;  /* Orient the mesh in parallel */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->meshNum = 0;
  options->orient  = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Parallel Mesh Orientation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mesh_num", "The test mesh number", "ex35.c", options->meshNum, &options->meshNum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-orient", "Orient the mesh in parallel", "ex35.c", options->orient, &options->orient, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh_0(MPI_Comm comm, PetscInt ornt, DM *dm)
{
  PetscSF        sf;
  PetscInt       Np[4]        = {4, 6, 4, 1};
  PetscInt       coneSize[15] = {4, 0, 0, 0, 0, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2};
  PetscInt       ornts[28];
  PetscInt       cones[28];
  PetscScalar    coord[12];
  PetscInt       locals[7];
  PetscSFNode    remotes[7];
  const PetscInt depth = 3;
  const PetscInt dim   = 3;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh 0 can only be constructed on 2 processes, not %d", size);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, &sf);CHKERRQ(ierr);
  switch (rank) {
    case 0:
    cones[0]  =  5;cones[1]  =  6;cones[2]  =  8;cones[3] = 7;
    ornts[0]  =  0;ornts[1]  =  0;ornts[2]  =  2;ornts[3] = 2;
    cones[4]  =  9;cones[5]  = 10;cones[6]  = 11; /* Vertices 1, 2, 3 */
    ornts[4]  =  0;ornts[5]  =  0;ornts[6]  =  0;
    cones[7]  = 12;cones[8]  = 13;cones[9]  =  9; /* Vertices 1, 4, 2 */
    ornts[7]  =  0;ornts[8]  = -2;ornts[9]  = -2;
    cones[10] = 13;cones[11] = 14;cones[12] = 10; /* Vertices 2, 4, 3 */
    ornts[10] =  0;ornts[11] = -2;ornts[12] = -2;
    cones[13] = 14;cones[14] = 12;cones[15] = 11; /* Vertices 3, 4, 1 */
    ornts[13] =  0;ornts[14] = -2;ornts[15] = -2;
    cones[16] =  1;cones[17] =  2;cones[18] =  2;cones[19] = 3;cones[20] = 3;cones[21] = 1;
    ornts[16] =  0;ornts[17] =  0;ornts[18] =  0;ornts[19] = 0;ornts[20] = 0;ornts[21] = 0;
    cones[22] =  1;cones[23] =  4;cones[24] =  2;cones[25] = 4;cones[26] = 3;cones[27] = 4;
    ornts[22] =  0;ornts[23] =  0;ornts[24] =  0;ornts[25] = 0;ornts[26] = 0;ornts[27] = 0;
    coord[0*dim+0] =  0.0;coord[0*dim+1] =  1.0;coord[0*dim+2] =  0.0;
    coord[1*dim+0] =  0.0;coord[1*dim+1] = -1.0;coord[1*dim+2] =  0.0;
    coord[2*dim+0] =  0.0;coord[2*dim+1] =  0.0;coord[2*dim+2] = -1.0;
    coord[3*dim+0] = -1.0;coord[3*dim+1] =  0.0;coord[3*dim+2] =  0.0;
    ierr = PetscSFSetGraph(sf, 15, 0, NULL, PETSC_COPY_VALUES, NULL, PETSC_COPY_VALUES);CHKERRQ(ierr);
    {
      const PetscInt cone[3] = {9, 10, 11};
      const PetscInt start   = ornt >= 0 ? -ornt : -(ornt+1);
      const PetscInt inc     = ornt >= 0 ? 1 : -1;
      const PetscInt fornt   = ornt >= 0 ? 0 : -2;

      /* Rotate face 5 */
      ornts[0] = ornt;
      cones[4] = cone[(start+3)%3];cones[5] = cone[(start+3+inc)%3];cones[6] = cone[(start+3+inc+inc)%3];
      ornts[4] = fornt;            ornts[5] = fornt;                ornts[6] = fornt;
    }
    break;
    case 1:
    cones[0]  =  5;cones[1]  =  6;cones[2]  =  8;cones[3] = 7;
    ornts[0]  =  0;ornts[1]  =  0;ornts[2]  =  2;ornts[3] = 2;
    cones[4]  =  9;cones[5]  = 10;cones[6]  = 11; /* Vertices 1, 2, 3 */
    ornts[4]  =  0;ornts[5]  =  0;ornts[6]  =  0;
    cones[7]  = 12;cones[8]  = 13;cones[9]  =  9; /* Vertices 1, 4, 2 */
    ornts[7]  =  0;ornts[8]  = -2;ornts[9]  = -2;
    cones[10] = 13;cones[11] = 14;cones[12] = 10; /* Vertices 2, 4, 3 */
    ornts[10] =  0;ornts[11] = -2;ornts[12] = -2;
    cones[13] = 14;cones[14] = 12;cones[15] = 11; /* Vertices 3, 4, 1 */
    ornts[13] =  0;ornts[14] = -2;ornts[15] = -2;
    cones[16] =  1;cones[17] =  2;cones[18] =  2;cones[19] = 3;cones[20] = 3;cones[21] = 1;
    ornts[16] =  0;ornts[17] =  0;ornts[18] =  0;ornts[19] = 0;ornts[20] = 0;ornts[21] = 0;
    cones[22] =  1;cones[23] =  4;cones[24] =  2;cones[25] = 4;cones[26] = 3;cones[27] = 4;
    ornts[22] =  0;ornts[23] =  0;ornts[24] =  0;ornts[25] = 0;ornts[26] = 0;ornts[27] = 0;
    coord[0*dim+0] =  0.0;coord[0*dim+1] = -1.0;coord[0*dim+2] =  0.0;
    coord[1*dim+0] =  0.0;coord[1*dim+1] =  1.0;coord[1*dim+2] =  0.0;
    coord[2*dim+0] =  0.0;coord[2*dim+1] =  0.0;coord[2*dim+2] = -1.0;
    coord[3*dim+0] =  1.0;coord[3*dim+1] =  0.0;coord[3*dim+2] =  0.0;
    locals[0] = 1;
    remotes[0].index =  2;remotes[0].rank = 0;
    locals[1] = 2;
    remotes[1].index =  1;remotes[1].rank = 0;
    locals[2] = 3;
    remotes[2].index =  3;remotes[2].rank = 0;
    locals[3] = 5;
    remotes[3].index =  5;remotes[3].rank = 0;
    locals[4] = 9;
    remotes[4].index =  9;remotes[4].rank = 0;
    locals[5] = 10;
    remotes[5].index = 11;remotes[5].rank = 0;
    locals[6] = 11;
    remotes[6].index = 10;remotes[6].rank = 0;
    ierr = PetscSFSetGraph(sf, 15, 7, locals, PETSC_COPY_VALUES, remotes, PETSC_COPY_VALUES);CHKERRQ(ierr);
    break;
  }
  ierr = DMPlexCreateFromDAG(*dm, depth, Np, coneSize, cones, ornts, coord);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dm, sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  {
    PetscInt *closure = NULL;
    PetscInt  clSize, cl;

    ierr = DMPlexGetTransitiveClosure(*dm, 5, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Face 5, ornt %D\n  ", ornt);CHKERRQ(ierr);
    for (cl = 0; cl < clSize; ++cl) {
      ierr = PetscPrintf(comm, " (%D, %D)", closure[cl*2], closure[cl*2+1]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
    ierr = DMPlexRestoreTransitiveClosure(*dm, 5, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);

    ierr = DMPlexGetTransitiveClosure(*dm, 0, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Cell 0\n  ");CHKERRQ(ierr);
    for (cl = 0; cl < clSize; ++cl) {
      ierr = PetscPrintf(comm, " (%D, %D)", closure[cl*2], closure[cl*2+1]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
    ierr = DMPlexRestoreTransitiveClosure(*dm, 0, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh_1(MPI_Comm comm, DM *dm)
{
  PetscSF        sf;
  PetscInt       Np[4]        = {4, 6, 4, 1};
  PetscInt       coneSize[15] = {4, 0, 0, 0, 0, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2};
  PetscInt       ornts[28];
  PetscInt       cones[28];
  PetscScalar    coord[12];
  PetscInt       locals[7];
  PetscSFNode    remotes[7];
  const PetscInt depth = 3;
  const PetscInt dim   = 3;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size != 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh 1 can only be constructed on 3 processes, not %d", size);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, &sf);CHKERRQ(ierr);
  switch (rank) {
    case 0:
    cones[0]  =  5;cones[1]  =  6;cones[2]  =  8;cones[3] = 7;
    ornts[0]  =  0;ornts[1]  =  0;ornts[2]  =  2;ornts[3] = 2;
    cones[4]  =  9;cones[5]  = 10;cones[6]  = 11; /* Vertices 1, 2, 3 */
    ornts[4]  =  0;ornts[5]  =  0;ornts[6]  =  0;
    cones[7]  = 12;cones[8]  = 13;cones[9]  =  9; /* Vertices 1, 4, 2 */
    ornts[7]  =  0;ornts[8]  = -2;ornts[9]  = -2;
    cones[10] = 13;cones[11] = 14;cones[12] = 10; /* Vertices 2, 4, 3 */
    ornts[10] =  0;ornts[11] = -2;ornts[12] = -2;
    cones[13] = 14;cones[14] = 12;cones[15] = 11; /* Vertices 3, 4, 1 */
    ornts[13] =  0;ornts[14] = -2;ornts[15] = -2;
    cones[16] =  1;cones[17] =  2;cones[18] =  2;cones[19] = 3;cones[20] = 3;cones[21] = 1;
    ornts[16] =  0;ornts[17] =  0;ornts[18] =  0;ornts[19] = 0;ornts[20] = 0;ornts[21] = 0;
    cones[22] =  1;cones[23] =  4;cones[24] =  2;cones[25] = 4;cones[26] = 3;cones[27] = 4;
    ornts[22] =  0;ornts[23] =  0;ornts[24] =  0;ornts[25] = 0;ornts[26] = 0;ornts[27] = 0;
    coord[0*dim+0] =  0.0;coord[0*dim+1] =  1.0;coord[0*dim+2] =  0.0;
    coord[1*dim+0] =  0.0;coord[1*dim+1] = -1.0;coord[1*dim+2] =  0.0;
    coord[2*dim+0] =  0.0;coord[2*dim+1] =  0.0;coord[2*dim+2] = -1.0;
    coord[3*dim+0] = -1.0;coord[3*dim+1] =  0.0;coord[3*dim+2] =  0.0;
    ierr = PetscSFSetGraph(sf, 15, 0, NULL, PETSC_COPY_VALUES, NULL, PETSC_COPY_VALUES);CHKERRQ(ierr);
    locals[0] = 9;
    remotes[0].index = 9;remotes[0].rank = 2;
    ierr = PetscSFSetGraph(sf, 15, 1, locals, PETSC_COPY_VALUES, remotes, PETSC_COPY_VALUES);CHKERRQ(ierr);
    break;
    case 1:
    cones[0]  =  5;cones[1]  =  6;cones[2]  =  8;cones[3] = 7;
    ornts[0]  =  0;ornts[1]  =  0;ornts[2]  =  2;ornts[3] = 2;
    cones[4]  =  9;cones[5]  = 10;cones[6]  = 11; /* Vertices 1, 2, 3 */
    ornts[4]  =  0;ornts[5]  =  0;ornts[6]  =  0;
    cones[7]  = 12;cones[8]  = 13;cones[9]  =  9; /* Vertices 1, 4, 2 */
    ornts[7]  =  0;ornts[8]  = -2;ornts[9]  = -2;
    cones[10] = 13;cones[11] = 14;cones[12] = 10; /* Vertices 2, 4, 3 */
    ornts[10] =  0;ornts[11] = -2;ornts[12] = -2;
    cones[13] = 14;cones[14] = 12;cones[15] = 11; /* Vertices 3, 4, 1 */
    ornts[13] =  0;ornts[14] = -2;ornts[15] = -2;
    cones[16] =  1;cones[17] =  2;cones[18] =  2;cones[19] = 3;cones[20] = 3;cones[21] = 1;
    ornts[16] =  0;ornts[17] =  0;ornts[18] =  0;ornts[19] = 0;ornts[20] = 0;ornts[21] = 0;
    cones[22] =  1;cones[23] =  4;cones[24] =  2;cones[25] = 4;cones[26] = 3;cones[27] = 4;
    ornts[22] =  0;ornts[23] =  0;ornts[24] =  0;ornts[25] = 0;ornts[26] = 0;ornts[27] = 0;
    coord[0*dim+0] =  0.0;coord[0*dim+1] = -1.0;coord[0*dim+2] =  0.0;
    coord[1*dim+0] =  0.0;coord[1*dim+1] =  1.0;coord[1*dim+2] =  0.0;
    coord[2*dim+0] =  0.0;coord[2*dim+1] =  0.0;coord[2*dim+2] = -1.0;
    coord[3*dim+0] =  1.0;coord[3*dim+1] =  0.0;coord[3*dim+2] =  0.0;
    locals[0] = 1;
    remotes[0].index =  2;remotes[0].rank = 0;
    locals[1] = 2;
    remotes[1].index =  1;remotes[1].rank = 0;
    locals[2] = 3;
    remotes[2].index =  3;remotes[2].rank = 0;
    locals[3] = 5;
    remotes[3].index =  5;remotes[3].rank = 0;
    locals[4] = 9;
    remotes[4].index =  9;remotes[4].rank = 2;
    locals[5] = 10;
    remotes[5].index = 11;remotes[5].rank = 0;
    locals[6] = 11;
    remotes[6].index = 10;remotes[6].rank = 0;
    ierr = PetscSFSetGraph(sf, 15, 7, locals, PETSC_COPY_VALUES, remotes, PETSC_COPY_VALUES);CHKERRQ(ierr);
    break;
    case 2:
    cones[0]  =  5;cones[1]  =  6;cones[2]  =  8;cones[3] = 7;
    ornts[0]  =  0;ornts[1]  =  0;ornts[2]  =  2;ornts[3] = 2;
    cones[4]  =  9;cones[5]  = 10;cones[6]  = 11; /* Vertices 1, 2, 3 */
    ornts[4]  =  0;ornts[5]  =  0;ornts[6]  =  0;
    cones[7]  = 12;cones[8]  = 13;cones[9]  =  9; /* Vertices 1, 4, 2 */
    ornts[7]  =  0;ornts[8]  = -2;ornts[9]  = -2;
    cones[10] = 13;cones[11] = 14;cones[12] = 10; /* Vertices 2, 4, 3 */
    ornts[10] =  0;ornts[11] = -2;ornts[12] = -2;
    cones[13] = 14;cones[14] = 12;cones[15] = 11; /* Vertices 3, 4, 1 */
    ornts[13] =  0;ornts[14] = -2;ornts[15] = -2;
    cones[16] =  1;cones[17] =  2;cones[18] =  2;cones[19] = 3;cones[20] = 3;cones[21] = 1;
    ornts[16] =  0;ornts[17] =  0;ornts[18] =  0;ornts[19] = 0;ornts[20] = 0;ornts[21] = 0;
    cones[22] =  1;cones[23] =  4;cones[24] =  2;cones[25] = 4;cones[26] = 3;cones[27] = 4;
    ornts[22] =  0;ornts[23] =  0;ornts[24] =  0;ornts[25] = 0;ornts[26] = 0;ornts[27] = 0;
    coord[0*dim+0] =  0.0;coord[0*dim+1] = -1.0;coord[0*dim+2] =  0.0;
    coord[1*dim+0] =  0.0;coord[1*dim+1] =  1.0;coord[1*dim+2] =  0.0;
    coord[2*dim+0] =  1.0;coord[2*dim+1] =  0.0;coord[2*dim+2] =  0.0;
    coord[3*dim+0] =  1.0;coord[3*dim+1] =  0.0;coord[3*dim+2] =  1.0;
    locals[0] = 1;
    remotes[0].index =  2;remotes[0].rank = 0;
    locals[1] = 2;
    remotes[1].index =  1;remotes[1].rank = 0;
    locals[2] = 3;
    remotes[2].index =  4;remotes[2].rank = 1;
    locals[3] = 5;
    remotes[3].index =  6;remotes[3].rank = 1;
    locals[4] = 10;
    remotes[4].index = 13;remotes[4].rank = 1;
    locals[5] = 11;
    remotes[5].index = 12;remotes[5].rank = 1;
    ierr = PetscSFSetGraph(sf, 15, 6, locals, PETSC_COPY_VALUES, remotes, PETSC_COPY_VALUES);CHKERRQ(ierr);
    break;
  }
  ierr = DMPlexCreateFromDAG(*dm, depth, Np, coneSize, cones, ornts, coord);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dm, sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, PetscInt meshNum, PetscInt ornt, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(meshNum) {
    case 0: CreateMesh_0(comm, ornt, dm); break;
    case 1: CreateMesh_1(comm, dm); break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid test mesh number %D", meshNum);
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckMesh(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSkeleton(dm, 0);CHKERRQ(ierr);
  ierr = DMPlexCheckFaces(dm, 0);CHKERRQ(ierr);
  ierr = DMPlexCheckGeometry(dm);CHKERRQ(ierr);

  /* Waiting for fix ierr = DMPlexCheckPointSF(dm);CHKERRQ(ierr);*/
  ierr = DMPlexCheckConesConformOnInterfaces(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  if (ctx.meshNum == 0) {
    PetscInt ornt;

    for (ornt = -3; ornt < 3; ++ornt) {
      ierr = CreateMesh(PETSC_COMM_WORLD, ctx.meshNum, ornt, &dm);CHKERRQ(ierr);
      if (ctx.orient) {ierr = DMPlexOrient(dm);CHKERRQ(ierr);}
      ierr = CheckMesh(dm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
    }
  } else if (ctx.meshNum == 1) {
    ierr = CreateMesh(PETSC_COMM_WORLD, ctx.meshNum, 0, &dm);CHKERRQ(ierr);
    if (ctx.orient) {ierr = DMPlexOrient(dm);CHKERRQ(ierr);}
    ierr = CheckMesh(dm);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    nsize: 2
    args: -dm_view ::ascii_info_detail -ornt_dm_view ::ascii_info_detail -dm_plex_print_orient

  test:
    suffix: 1
    nsize: 3
    args: -mesh_num 1 -dm_view ::ascii_info_detail -ornt_dm_view ::ascii_info_detail -dm_plex_print_orient
TEST*/
