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

static PetscErrorCode CreateMesh_0(MPI_Comm comm, DM *dm)
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

static PetscErrorCode CreateMesh(MPI_Comm comm, PetscInt meshNum, DM *dm)
{
  PetscFunctionBegin;
  switch(meshNum) {
    case 0: CreateMesh_0(comm, dm); break;
    case 1: CreateMesh_1(comm, dm); break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid test mesh number %D", meshNum);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeRelativeOrientation(PetscInt n, const PetscInt ccone[], const PetscInt cone[], PetscInt *rornt)
{
  PetscInt c0, c1, c, d;

  PetscFunctionBegin;
  *rornt = 0;
  if (n <= 1) PetscFunctionReturn(0);
  /* Find first cone point in canonical array */
  c0 = cone[0];
  c1 = cone[1];
  for (c = 0; c < n; ++c) if (c0 == ccone[c]) break;
  if (c == n) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Initial cone point %D not found in canonical cone", c0);
  /* Check direction for iteration */
  if (c1 == ccone[(c+1)%n]) {
    /* Forward */
    for (d = 0; d < n; ++d) if (ccone[d] != cone[(c+d)%n]) break;
    if (d < n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Failed to compute relative cone orientation");
    *rornt = c;
  } else {
    /* Reverse */
    for (d = 0; d < n; ++d) if (ccone[d] != cone[(c+n-d)%n]) break;
    if (d < n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Failed to compute relative cone orientation");
    *rornt = -(c+1);
  }
  PetscFunctionReturn(0);
}

static PetscInt CanonicalOrientation(PetscInt n, PetscInt o)
{
  if (n == 2) {
    switch (o) {
      case 1:
        return -2;
      case -1:
        return 0;
      default: return o;
    }
  }
  return o;
}

/* Need a theory for composition
  - Not simple because we have to reproduce the eventual vertex order
*/
static PetscErrorCode ComposeOrientation(PetscInt n, PetscInt rornt, PetscInt oornt, PetscInt *nornt)
{
  const PetscInt pornt = oornt >= 0 ? oornt : -(oornt+1);
  PetscInt       pnornt;

  PetscFunctionBegin;
  if (rornt >= 0) {
    pnornt = (pornt + rornt)%n;
  } else {
    pnornt = (pornt - (rornt+1))%n;
  }
  *nornt = CanonicalOrientation(n, (oornt >= 0 && rornt >= 0) || (oornt < 0 && rornt < 0) ? pnornt : -(pnornt+1));
  PetscFunctionReturn(0);
}

/*
  DMPlexOrientSharedCones_Internal - Make the orientation of shared cones consistent across processes by changing the orientation of unowned points

  Collective on dm

  Input Parameters:
+ dm     - The DM
. ccones - An array of cones in canonical order, with layout given by the Plex cone section
- cornts - An array of point orientations for the canonical cones, with layout given by the Plex cone section

  Level: developer

.seealso: DMPlexGetConeSection(), DMGetPointSF()
*/
static PetscErrorCode DMPlexOrientSharedCones_Internal(DM dm, PetscInt depth, const PetscInt ccones[], const PetscInt cornts[])
{
  MPI_Comm           comm;
  PetscSection       s;
  PetscSF            sf;
  DMLabel            depthLabel;
  const PetscInt    *local;
  const PetscSFNode *remote;
  PetscInt          *nornt;
  PetscInt           maxConeSize, Nr, Nl, l;
  PetscMPIInt        rank;
  PetscBool          printOrient = PETSC_FALSE;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_print_orient", &printOrient, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dm, &s);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &Nr, &Nl, &local, &remote);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxConeSize, &nornt);CHKERRQ(ierr);
  /* Loop through leaves in order of depth */
    for (l = 0; l < Nl; ++l) {
      const PetscInt  point = local[l];
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, c, dep, off, rornt;

      ierr = DMLabelGetValue(depthLabel, point, &dep);CHKERRQ(ierr);
      if (dep != depth) continue;
      if (printOrient) {ierr = PetscSynchronizedPrintf(comm, "[%d]Checking point %D\n", rank, point);CHKERRQ(ierr);}
      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, point, &ornt);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(s, point, &off);CHKERRQ(ierr);
      ierr = ComputeRelativeOrientation(coneSize, &ccones[off], cone, &rornt);CHKERRQ(ierr);
      if (rornt) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        if (printOrient) {
          ierr = PetscSynchronizedPrintf(comm, "[%d]  Fixing cone of point %D\n", rank, point);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "[%d]    Changed cone for point %D from (", rank, point);
          for (c = 0; c < coneSize; ++c) {
            if (c > 0) {ierr = PetscSynchronizedPrintf(comm, ", ");}
            ierr = PetscSynchronizedPrintf(comm, "%D/%D", cone[c], ornt[c]);
          }
          ierr = PetscSynchronizedPrintf(comm, ") to (");
          for (c = 0; c < coneSize; ++c) {
            if (c > 0) {ierr = PetscSynchronizedPrintf(comm, ", ");}
            ierr = PetscSynchronizedPrintf(comm, "%D/%D", ccones[off+c], cornts[off+c]);
          }
          ierr = PetscSynchronizedPrintf(comm, ")\n");
        }
        ierr = DMPlexSetCone(dm, point, &ccones[off]);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(dm, point, &cornts[off]);CHKERRQ(ierr);
        if (printOrient) {ierr = PetscSynchronizedPrintf(comm, "[%d]  Fixing orientation of point %D in mesh\n", rank, point);CHKERRQ(ierr);}
        ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          const PetscInt  spoint = support[s];
          const PetscInt *scone, *sornt;
          PetscInt        sconeSize, sc;

          ierr = DMPlexGetConeSize(dm, spoint, &sconeSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, spoint, &scone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, spoint, &sornt);CHKERRQ(ierr);
          for (sc = 0; sc < sconeSize; ++sc) if (scone[sc] == point) break;
          if (sc == sconeSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D not found in cone of %D", point, spoint);
          ierr = PetscArraycpy(nornt, sornt, sconeSize);CHKERRQ(ierr);
          ierr = ComposeOrientation(coneSize, rornt, sornt[sc], &nornt[sc]);CHKERRQ(ierr);
          if (printOrient) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Changed orientation for point %D in %D from %D to %D (%D)\n", rank, point, spoint, sornt[sc], nornt[sc], rornt);CHKERRQ(ierr);}
          ierr = DMPlexSetConeOrientation(dm, spoint, nornt);CHKERRQ(ierr);
        }
      }
    }
  if (printOrient) {ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);}
  ierr = PetscFree(nornt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionISViewFromOptions_Internal(PetscSection s, PetscBool isSFNode, const PetscInt a[], const char name[], const char opt[])
{
  IS             pIS;
  PetscInt       N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetStorageSize(s, &N);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) s), N * (isSFNode ? 2 : 1), a, PETSC_USE_POINTER, &pIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) pIS, name);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) pIS, NULL, opt);CHKERRQ(ierr);
  ierr = ISDestroy(&pIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  PetscSFPointsLocalToGlobal_Internal - Translate an array of local mesh point numbers to pairs (rank, point) where rank is the global owner

  Not collective

  Input Parameters:
+ sf     - The SF determining point ownership
. gs     - The global section giving the layout of mesh points array
- points - An array of mesh point numbers

  Output Parameter:
. gpoints - An array of pairs (rank, point) where rank is the global point owner

  Level: Developer

.seealso: PetscSFPointsGlobalToLocal_Internal(), DMPlexOrientParallel_Internal()
*/
static PetscErrorCode PetscSFPointsLocalToGlobal_Internal(PetscSF sf, PetscSection gs, const PetscInt points[], PetscSFNode gpoints[])
{
  PetscLayout        layout;
  const PetscSFNode *remote;
  const PetscInt    *local;
  PetscMPIInt        rank;
  PetscInt           N, Nl, pStart, pEnd, p, gStart;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) gs), &rank);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(gs, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(gs, &N);CHKERRQ(ierr);
  ierr = PetscSectionGetValueLayout(PetscObjectComm((PetscObject) gs), gs, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(layout, &gStart, NULL);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &Nl, &local, &remote);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, d, loc;

    ierr = PetscSectionGetDof(gs, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(gs, p, &off);CHKERRQ(ierr);
    if (dof < 0) continue;
    for (d = off; d < off+dof; ++d) {
      const PetscInt coff = d - gStart;

      if (coff >= N) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid index %D > %D for point %D (%D, %D)", coff, N, p, dof, off);
      ierr = PetscFindInt(points[coff], Nl, local, &loc);CHKERRQ(ierr);
      if (loc < 0) {
        gpoints[coff].index = points[coff];
        gpoints[coff].rank  = rank;
      } else {
        gpoints[coff].index = remote[loc].index;
        gpoints[coff].rank  = remote[loc].rank;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  PetscSFPointsGlobalToLocal_Internal - Translate an array of pairs (rank, point) where rank is the global owner to local mesh point numbers

  Not collective

  Input Parameters:
+ sf      - The SF determining point ownership
. s       - The local section giving the layout of mesh points array
- gpoints - An array of pairs (rank, point) where rank is the global point owner

  Output Parameter:
. points - An array of local mesh point numbers

  Level: Developer

.seealso: PetscSFPointsLocalToGlobal_Internal(), DMPlexOrientParallel_Internal()
*/
static PetscErrorCode PetscSFPointsGlobalToLocal_Internal(PetscSF sf, PetscSection s, const PetscSFNode gpoints[], PetscInt points[])
{
  const PetscSFNode *remote;
  const PetscInt    *local;
  PetscInt           Nl, l, m;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) sf), &rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &Nl, &local, &remote);CHKERRQ(ierr);
  for (l = 0; l < Nl; ++l) {
    PetscInt dof, off, d;

    ierr = PetscSectionGetDof(s, local[l], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(s, local[l], &off);CHKERRQ(ierr);
    for (d = off; d < off+dof; ++d) {
      const PetscSFNode rem = gpoints[d];

      if (rem.rank == rank) {
        points[d] = rem.index;
      } else {
        /* TODO Expand Petsc Sort/Find to SFNode */
        for (m = 0; m < Nl; ++m) if (remote[m].index == rem.index && remote[m].rank == rem.rank) break;
        if (m < Nl) points[d] = local[m];
        else SETERRQ6(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Remote point (%D, %D) for leaf %D (%D) not found in point SF (%D, %D)", rem.index, rem.rank, l, local[l], dof, off);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexOrientParallel_Internal - Given a mesh with consistent local orientation, construct a consistent global orientation

  Collective on DM

  Input Parameter:
. dm - The DM

  Level: developer

.seealso: DMPlexOrient(), DMGetPointSF()
*/
static PetscErrorCode DMPlexOrientParallel_Internal(DM dm)
{
  PetscSF         sf, csf;
  PetscSection    s, gs;
  const PetscInt *cones, *ornts;
  PetscInt       *gcones, *ccones, *gornts, *cornts, *remoteOffsets;
  PetscSFNode    *rgcones, *rccones;
  PetscInt        depth, d, Nc, gNc;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dm, &s);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientations(dm, &ornts);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) s, NULL, "-cone_section_view");CHKERRQ(ierr);
  /* Create global section and section SF for cones */
  ierr = PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &gs);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) gs, "Global Cone Section");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) gs, NULL, "-cone_global_section_view");CHKERRQ(ierr);
  {
    PetscSection ts;
    PetscLayout  layout;
    PetscInt     pStart, pEnd, p, gStart;

    ierr = PetscSectionClone(gs, &ts);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(ts, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetValueLayout(PetscObjectComm((PetscObject) ts), ts, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(layout, &gStart, NULL);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt off;

      ierr = PetscSectionGetOffset(ts, p, &off);CHKERRQ(ierr);
      if (off >= 0) {ierr = PetscSectionSetOffset(ts, p,  off-gStart);CHKERRQ(ierr);}
    }
    ierr = PetscSFCreateRemoteOffsets(sf, ts, s, &remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&ts);CHKERRQ(ierr);
  }
  {
    IS       pIS;
    PetscInt pStart, pEnd;

    ierr = PetscSectionGetChart(gs, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) gs), pEnd-pStart, remoteOffsets, PETSC_USE_POINTER, &pIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) pIS, "Remote Offsets");CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) pIS, NULL, "-remote_offsets_view");CHKERRQ(ierr);
    ierr = ISDestroy(&pIS);CHKERRQ(ierr);
  }
  ierr = PetscSFCreateSectionSF(sf, gs, remoteOffsets, s, &csf);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) csf, "Cone SF");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) csf, NULL, "-cone_sf_view");CHKERRQ(ierr);
  /**/
  ierr = PetscSectionGetStorageSize(s, &Nc);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(gs, &gNc);CHKERRQ(ierr);
  ierr = PetscMalloc3(Nc, &ccones, Nc, &rccones, Nc, &cornts);CHKERRQ(ierr);
  ierr = PetscMalloc3(gNc, &gcones, gNc, &rgcones, gNc, &gornts);CHKERRQ(ierr);
  for (d = 0; d < depth; ++d) {
    ierr = PetscSFLocalToGlobalBegin(csf, s, gs, MPIU_INT, cones, INSERT_VALUES, gcones);CHKERRQ(ierr);
    ierr = PetscSFLocalToGlobalBegin(csf, s, gs, MPIU_INT, ornts, INSERT_VALUES, gornts);CHKERRQ(ierr);
    ierr = PetscSFLocalToGlobalEnd(csf, s, gs, MPIU_INT, cones, INSERT_VALUES, gcones);CHKERRQ(ierr);
    ierr = PetscSFLocalToGlobalEnd(csf, s, gs, MPIU_INT, ornts, INSERT_VALUES, gornts);CHKERRQ(ierr);
    ierr = PetscSFPointsLocalToGlobal_Internal(sf, gs, gcones, rgcones);CHKERRQ(ierr);
    ierr = PetscSectionISViewFromOptions_Internal(gs, PETSC_TRUE, (PetscInt *) rgcones, "Remote Global Cones", "-remote_global_cone_view");CHKERRQ(ierr);

    ierr = PetscSFGlobalToLocalBegin(csf, s, gs, MPIU_2INT, rgcones, INSERT_VALUES, rccones);CHKERRQ(ierr);
    ierr = PetscSFGlobalToLocalBegin(csf, s, gs, MPIU_INT, gornts, INSERT_VALUES, cornts);CHKERRQ(ierr);
    ierr = PetscSFGlobalToLocalEnd(csf, s, gs, MPIU_2INT, rgcones, INSERT_VALUES, rccones);CHKERRQ(ierr);
    ierr = PetscSFGlobalToLocalEnd(csf, s, gs, MPIU_INT, gornts, INSERT_VALUES, cornts);CHKERRQ(ierr);
    ierr = PetscSectionISViewFromOptions_Internal(s, PETSC_TRUE, (PetscInt *) rccones, "Remote Canonical Cones", "-remote_canonical_cone_view");CHKERRQ(ierr);

    ierr = PetscSFPointsGlobalToLocal_Internal(sf, s, rccones, ccones);CHKERRQ(ierr);
    ierr = PetscSectionISViewFromOptions_Internal(s, PETSC_FALSE, ccones, "Canonical Cones", "-canonical_cone_view");CHKERRQ(ierr);
    ierr = PetscSectionISViewFromOptions_Internal(s, PETSC_FALSE, ccones, "Canonical Cone Orientations", "-canonical_ornt_view");CHKERRQ(ierr);

    ierr = DMPlexOrientSharedCones_Internal(dm, d, ccones, cornts);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&gs);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&csf);CHKERRQ(ierr);
  ierr = PetscFree3(gcones, rgcones, gornts);CHKERRQ(ierr);
  ierr = PetscFree3(ccones, rccones, cornts);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-ornt_dm_view");CHKERRQ(ierr);
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

  //ierr = DMPlexCheckPointSF(dm);CHKERRQ(ierr);
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
  ierr = CreateMesh(PETSC_COMM_WORLD, ctx.meshNum, &dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  if (ctx.orient) {ierr = DMPlexOrientParallel_Internal(dm);CHKERRQ(ierr);}
  ierr = CheckMesh(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
