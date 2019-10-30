#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

/*@
  DMPlexGetRelativeOrientation - Compare the cone of the given mesh point with the given canonical cone (with the same cone points modulo order), and return the relative orientation.

  Not Collective

  Input Parameters:
+ dm     - The DM (DMPLEX)
. n      - The cone size
. ccone  - The canonical cone
- cone   - The cone

  Output Parameters:
. rornt  - The orientation that 'cone' would have to have in order to produce the canonical cone order when traversed

  Level: advanced

.seealso: DMPlexOrient()
@*/
PetscErrorCode DMPlexGetRelativeOrientation(DM dm, PetscInt n, const PetscInt ccone[], const PetscInt cone[], PetscInt *rornt)
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
    for (d = 0; d < n; ++d) if (cone[d] != ccone[(c+d)%n]) break;
    if (d < n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Failed to compute relative cone orientation");
    *rornt = c;
  } else {
    /* Reverse */
    for (d = 0; d < n; ++d) if (cone[d] != ccone[(c+n-d)%n]) break;
    if (d < n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Failed to compute relative cone orientation");
    *rornt = -(c+1);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexReverseCell - Give a mesh cell the opposite orientation

  Input Parameters:
+ dm   - The DM
- cell - The cell number

  Note: The modification of the DM is done in-place.

  Level: advanced

.seealso: DMPlexOrient(), DMCreate(), DMPLEX
@*/
PetscErrorCode DMPlexReverseCell(DM dm, PetscInt cell)
{
  /* Note that the reverse orientation ro of a face with orientation o is:

       ro = o >= 0 ? -(faceSize - o) : faceSize + o

     where faceSize is the size of the cone for the face.
  */
  const PetscInt *cone,    *coneO, *support;
  PetscInt       *revcone, *revconeO;
  PetscInt        maxConeSize, coneSize, supportSize, faceSize, cp, sp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, MPIU_INT, &revcone);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, MPIU_INT, &revconeO);CHKERRQ(ierr);
  /* Reverse cone, and reverse orientations of faces */
  ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, cell, &coneO);CHKERRQ(ierr);
  for (cp = 0; cp < coneSize; ++cp) {
    const PetscInt rcp = coneSize-cp-1;

    ierr = DMPlexGetConeSize(dm, cone[rcp], &faceSize);CHKERRQ(ierr);
    revcone[cp]  = cone[rcp];
    revconeO[cp] = coneO[rcp] >= 0 ? -(faceSize-coneO[rcp]) : faceSize+coneO[rcp];
  }
  ierr = DMPlexSetCone(dm, cell, revcone);CHKERRQ(ierr);
  ierr = DMPlexSetConeOrientation(dm, cell, revconeO);CHKERRQ(ierr);
  /* Reverse orientation of this cell in the support hypercells */
  faceSize = coneSize;
  ierr = DMPlexGetSupportSize(dm, cell, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, cell, &support);CHKERRQ(ierr);
  for (sp = 0; sp < supportSize; ++sp) {
    ierr = DMPlexGetConeSize(dm, support[sp], &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, support[sp], &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, support[sp], &coneO);CHKERRQ(ierr);
    for (cp = 0; cp < coneSize; ++cp) {
      if (cone[cp] != cell) continue;
      ierr = DMPlexInsertConeOrientation(dm, support[sp], cp, coneO[cp] >= 0 ? -(faceSize-coneO[cp]) : faceSize+coneO[cp]);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreWorkArray(dm, maxConeSize, MPIU_INT, &revcone);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, maxConeSize, MPIU_INT, &revconeO);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  - Checks face match
    - Flips non-matching
  - Inserts faces of support cells in FIFO
*/
static PetscErrorCode DMPlexCheckFace_Internal(DM dm, PetscInt *faceFIFO, PetscInt *fTop, PetscInt *fBottom, PetscInt cStart, PetscInt fStart, PetscInt fEnd, PetscBT seenCells, PetscBT flippedCells, PetscBT seenFaces)
{
  const PetscInt *support, *coneA, *coneB, *coneOA, *coneOB;
  PetscInt        supportSize, coneSizeA, coneSizeB, posA = -1, posB = -1;
  PetscInt        face, dim, seenA, flippedA, seenB, flippedB, mismatch, c;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  face = faceFIFO[(*fTop)++];
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);
  if (supportSize < 2) PetscFunctionReturn(0);
  if (supportSize != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Faces should separate only two cells, not %d", supportSize);
  seenA    = PetscBTLookup(seenCells,    support[0]-cStart);
  flippedA = PetscBTLookup(flippedCells, support[0]-cStart) ? 1 : 0;
  seenB    = PetscBTLookup(seenCells,    support[1]-cStart);
  flippedB = PetscBTLookup(flippedCells, support[1]-cStart) ? 1 : 0;

  ierr = DMPlexGetConeSize(dm, support[0], &coneSizeA);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, support[1], &coneSizeB);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, support[0], &coneA);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, support[1], &coneB);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, support[0], &coneOA);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, support[1], &coneOB);CHKERRQ(ierr);
  for (c = 0; c < coneSizeA; ++c) {
    if (!PetscBTLookup(seenFaces, coneA[c]-fStart)) {
      faceFIFO[(*fBottom)++] = coneA[c];
      ierr = PetscBTSet(seenFaces, coneA[c]-fStart);CHKERRQ(ierr);
    }
    if (coneA[c] == face) posA = c;
    if (*fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], *fBottom, fEnd-fStart);
  }
  if (posA < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d could not be located in cell %d", face, support[0]);
  for (c = 0; c < coneSizeB; ++c) {
    if (!PetscBTLookup(seenFaces, coneB[c]-fStart)) {
      faceFIFO[(*fBottom)++] = coneB[c];
      ierr = PetscBTSet(seenFaces, coneB[c]-fStart);CHKERRQ(ierr);
    }
    if (coneB[c] == face) posB = c;
    if (*fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], *fBottom, fEnd-fStart);
  }
  if (posB < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d could not be located in cell %d", face, support[1]);

  if (dim == 1) {
    mismatch = posA == posB;
  } else {
    mismatch = coneOA[posA] == coneOB[posB];
  }

  if (mismatch ^ (flippedA ^ flippedB)) {
    if (seenA && seenB) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen cells %d and %d do not match: Fault mesh is non-orientable", support[0], support[1]);
    if (!seenA && !flippedA) {
      ierr = PetscBTSet(flippedCells, support[0]-cStart);CHKERRQ(ierr);
    } else if (!seenB && !flippedB) {
      ierr = PetscBTSet(flippedCells, support[1]-cStart);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
  } else if (mismatch && flippedA && flippedB) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
  ierr = PetscBTSet(seenCells, support[0]-cStart);CHKERRQ(ierr);
  ierr = PetscBTSet(seenCells, support[1]-cStart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexOrient - Give a consistent orientation to the input mesh

  Input Parameters:
. dm - The DM

  Note: The orientation data for the DM are change in-place.
$ This routine will fail for non-orientable surfaces, such as the Moebius strip.

  Level: advanced

.seealso: DMCreate(), DMPLEX
@*/
PetscErrorCode DMPlexOrient(DM dm)
{
  PetscBT        seenCells, flippedCells, seenFaces;
  PetscInt      *faceFIFO, fTop, fBottom, *cellComp, *faceComp;
  PetscInt       dim, h, cStart, cEnd, c, cell, fStart, fEnd, face;
  PetscMPIInt    comp = 0;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-orientation_view", &flg);CHKERRQ(ierr);
  /* Truth Table
     mismatch    flips   do action   mismatch   flipA ^ flipB   action
         F       0 flips     no         F             F           F
         F       1 flip      yes        F             T           T
         F       2 flips     no         T             F           T
         T       0 flips     yes        T             T           F
         T       1 flip      no
         T       2 flips     yes
  */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &h);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h+1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = PetscBTCreate(cEnd - cStart, &seenCells);CHKERRQ(ierr);
  ierr = PetscBTMemzero(cEnd - cStart, seenCells);CHKERRQ(ierr);
  ierr = PetscBTCreate(cEnd - cStart, &flippedCells);CHKERRQ(ierr);
  ierr = PetscBTMemzero(cEnd - cStart, flippedCells);CHKERRQ(ierr);
  ierr = PetscBTCreate(fEnd - fStart, &seenFaces);CHKERRQ(ierr);
  ierr = PetscBTMemzero(fEnd - fStart, seenFaces);CHKERRQ(ierr);
  ierr = PetscCalloc3(fEnd - fStart, &faceFIFO, cEnd-cStart, &cellComp, fEnd-fStart, &faceComp);CHKERRQ(ierr);
  /* Loop over components */
  for (cell = cStart; cell < cEnd; ++cell) cellComp[cell-cStart] = -1;
  do {
    /* Look for first unmarked cell */
    for (cell = cStart; cell < cEnd; ++cell) if (cellComp[cell-cStart] < 0) break;
    if (cell >= cEnd) break;
    /* Initialize FIFO with first cell in component */
    {
      const PetscInt *cone;
      PetscInt        coneSize;

      fTop = fBottom = 0;
      ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        faceFIFO[fBottom++] = cone[c];
        ierr = PetscBTSet(seenFaces, cone[c]-fStart);CHKERRQ(ierr);
      }
      ierr = PetscBTSet(seenCells, cell-cStart);CHKERRQ(ierr);
    }
    /* Consider each face in FIFO */
    while (fTop < fBottom) {
      ierr = DMPlexCheckFace_Internal(dm, faceFIFO, &fTop, &fBottom, cStart, fStart, fEnd, seenCells, flippedCells, seenFaces);CHKERRQ(ierr);
    }
    /* Set component for cells and faces */
    for (cell = 0; cell < cEnd-cStart; ++cell) {
      if (PetscBTLookup(seenCells, cell)) cellComp[cell] = comp;
    }
    for (face = 0; face < fEnd-fStart; ++face) {
      if (PetscBTLookup(seenFaces, face)) faceComp[face] = comp;
    }
    /* Wipe seenCells and seenFaces for next component */
    ierr = PetscBTMemzero(fEnd - fStart, seenFaces);CHKERRQ(ierr);
    ierr = PetscBTMemzero(cEnd - cStart, seenCells);CHKERRQ(ierr);
    ++comp;
  } while (1);
  if (flg) {
    PetscViewer v;
    MPI_Comm    comm;
    PetscMPIInt rank;

    ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetStdout(comm, &v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for serial flipped cells:\n", rank);CHKERRQ(ierr);
    ierr = PetscBTView(cEnd-cStart, flippedCells, v);CHKERRQ(ierr);
    ierr = PetscViewerFlush(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(v);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&seenCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&flippedCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&seenFaces);CHKERRQ(ierr);
  ierr = PetscFree3(faceFIFO, cellComp, faceComp);CHKERRQ(ierr);

  ierr = DMPlexOrientInterface_Internal(dm);CHKERRQ(ierr);
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

/* Remove reduncant orientations */
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
      ierr = DMPlexGetRelativeOrientation(dm, coneSize, &ccones[off], cone, &rornt);CHKERRQ(ierr);
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

/*
  DMPlexOrientParallel_Internal - Given a mesh with consistent local orientation, construct a consistent global orientation

  Collective on DM

  Input Parameter:
. dm - The DM

  Level: developer

.seealso: DMPlexOrient(), DMGetPointSF()
*/
PetscErrorCode DMPlexOrientInterface_Internal(DM dm)
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
  ierr = PetscSFCreateRemoteOffsets(sf, gs, PETSC_TRUE, s, &remoteOffsets);CHKERRQ(ierr);
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
