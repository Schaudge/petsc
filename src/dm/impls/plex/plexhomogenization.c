#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petscsnes.h>

static PetscErrorCode PrintMatSetValues(PetscViewer viewer, Mat A, PetscInt point, PetscInt numRIndices, const PetscInt rindices[], PetscInt numCIndices, const PetscInt cindices[], const PetscScalar values[])
{
  PetscMPIInt rank;
  PetscInt    i, j;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]mat for point %" PetscInt_FMT "\n", rank, point));
  for (i = 0; i < numRIndices; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]mat row indices[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, i, rindices[i]));
  for (i = 0; i < numCIndices; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]mat col indices[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, i, cindices[i]));
  numCIndices = numCIndices ? numCIndices : numRIndices;
  if (!values) PetscFunctionReturn(PETSC_SUCCESS);
  for (i = 0; i < numRIndices; i++) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]", rank));
    for (j = 0; j < numCIndices; j++) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(viewer, " (%g,%g)", (double)PetscRealPart(values[i * numCIndices + j]), (double)PetscImaginaryPart(values[i * numCIndices + j])));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer, " %g", (double)values[i * numCIndices + j]));
#endif
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create an ephemeral refined mesh
static PetscErrorCode DMPlexRefineMesh_Private(DM dm, DM *rdm) {
  DMPlexTransform tr;
  const char     *name, *prefix;
  char            rname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "refine_"));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));

  PetscCall(DMPlexCreateEphemeral(tr, rdm));
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(PetscSNPrintf(rname, PETSC_MAX_PATH_LEN, "Ephemeral Refined %s", name));
  PetscCall(PetscObjectSetName((PetscObject)*rdm, rname));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*rdm, prefix));
  PetscCall(PetscObjectPrependOptionsPrefix((PetscObject)*rdm, "ref_"));
  PetscCall(DMViewFromOptions(*rdm, NULL, "-dm_view"));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMPlexSetRegularRefinement(*rdm, PETSC_TRUE));
  PetscCall(DMSetCoarseDM(*rdm, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TODO: Patch should be on PETSC_COMM_SELF
static PetscErrorCode DMPlexCreatePatch(DM dm, PetscInt cell, DM *pdm)
{
  DMPlexTransform tr;
  DMLabel         active;
  MPI_Comm        comm;
  PetscInt       *adj     = NULL;
  PetscInt        adjSize = PETSC_DETERMINE;
  const char     *name, *prefix;
  char            pname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMLabelCreate(comm, "active", &active));
  PetscCall(DMPlexGetAdjacency(dm, cell, &adjSize, &adj));
  for (PetscInt a = 0; a < adjSize; ++a) PetscCall(DMLabelSetValue(active, adj[a], DM_ADAPT_REFINE));
  PetscCall(PetscObjectViewFromOptions((PetscObject)active, NULL, "-active_view"));
  PetscCall(PetscFree(adj));

  PetscCall(DMPlexTransformCreate(comm, &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "select_"));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetActive(tr, active));
  PetscCall(DMPlexTransformSetType(tr, DMPLEXTRANSFORMFILTER));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));
  PetscCall(DMLabelDestroy(&active));

  PetscCall(DMPlexCreateEphemeral(tr, pdm));
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(PetscSNPrintf(pname, PETSC_MAX_PATH_LEN, "Ephemeral Patch %s", name));
  PetscCall(PetscObjectSetName((PetscObject)*pdm, pname));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*pdm, prefix));
  PetscCall(PetscObjectPrependOptionsPrefix((PetscObject)*pdm, "patch_"));
  PetscCall(DMViewFromOptions(*pdm, NULL, "-patch_view"));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMPlexSetLocationAlg(*pdm, DM_PLEX_LOCATE_GRID_HASH));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create global prolongator from dm to rdm
static PetscErrorCode DMPlexCreateProlongator_Private(DM dm, DM rdm, Mat *P)
{
  PetscSection gs, rgs;
  PetscInt     m, n;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalSection(dm, &gs));
  PetscCall(DMGetGlobalSection(rdm, &rgs));
  PetscCall(PetscSectionGetConstrainedStorageSize(rgs, &m));
  PetscCall(PetscSectionGetConstrainedStorageSize(gs, &n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), P));
  PetscCall(MatSetSizes(*P, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetUp(*P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {PATCH_SYS_IDENTITY, PATCH_SYS_LOD} PatchSystemType;
const char *PatchSystemTypes[] = {"identity", "lod", "PatchSystemType", "PATCH_SYS_", NULL};

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt c = 0; c < Nc; ++c) u[c] = 0.0;
  return(PETSC_SUCCESS);
}

static void f0_id_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0];
}

static void g0_id_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode SetupPatchProblem_Identity_Private(DM dm)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_id_phi, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_id_phi, NULL, NULL, NULL));
  if (label) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f1_lap_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_lap_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static PetscErrorCode SetupPatchProblem_LOD_Private(DM dm)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, f1_lap_phi));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_lap_phi));
  if (label) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, NULL, NULL));
  //PetscCall(PetscDSSetResidual(ds, 1, f0_mu, NULL));
  //PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_mumu, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization_Private(DM dm, PetscInt Nf, const char *names[], PetscErrorCode (*setup)(DM))
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", names[f]));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, prefix, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, names[f]));
    PetscCall(DMSetField(dm, f, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPatchProblem_Private(DM patch, PatchSystemType systype)
{
  const char *names[] = {"phi", "mu"};

  PetscFunctionBegin;
  switch (systype) {
  case PATCH_SYS_IDENTITY:
    PetscCall(SetupDiscretization_Private(patch, 1, names, SetupPatchProblem_Identity_Private));
    break;
  case PATCH_SYS_LOD:
    PetscCall(SetupDiscretization_Private(patch, 2, names, SetupPatchProblem_LOD_Private));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexPatchSolve - Solve the saddle-point system on the refined patch and inject the results into the corrector

  Input Parameters:
+ dm     - The coarse mesh over the whole domain
. patch  - The patch from the coarse grid
. c      - The central cell for this patch
. rdm    - The refined mesh over the whole domain
. rpatch - The patch from the fine grid
. P      - The global prolongator matrix
- user   - A user context

  Note:
  Coarse indices are those for the closure of the original seed cell. Fine indices are those for the closure of the entire refined patch, so we just indicate the whole section

  Level; advanced

.seealso: `CreatePatch()`
*/
static PetscErrorCode DMPlexPatchSolve(DM dm, DM patch, PetscInt c, DM rdm, DM rpatch, Mat P)
{
  SNES            snes;
  Mat             pP, pM;
  Vec             u, b, psi, cpsi;
  IS              subpIS;
  PetscScalar    *elemP;
  PetscSection    s, sRef, gs, gsRef, dgs, dgsRef;
  const PetscInt *points;
  PetscInt       *closure = NULL, *rows, *cols;
  PetscInt        cell, pStart, pEnd, Ncl, Nfine = 0, Ncoarse = 0, j = 0;
  char            name[PETSC_MAX_PATH_LEN];
  PetscBool       viewPatchSol = PETSC_FALSE;
  void           *ctx = NULL;

  PetscFunctionBegin;
  PetscCall(SetupPatchProblem_Private(patch, PATCH_SYS_IDENTITY));
  PetscCall(SetupPatchProblem_Private(rpatch, PATCH_SYS_IDENTITY));
  { // Check that the patch contains cell c
    PetscInt n;

    PetscCall(DMPlexGetSubpointIS(patch, &subpIS));
    PetscCheck(subpIS, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Input patch mesh has no connection to the domain mesh (subpoint IS)");
    PetscCall(PetscObjectViewFromOptions((PetscObject)subpIS, NULL, "-subpoint_is_view"));
    PetscCall(ISGetLocalSize(subpIS, &n));
    PetscCall(ISGetIndices(subpIS, &points));
    for (cell = 0; cell < n; ++cell) if (points[cell] == c) break;
    PetscCheck(cell < n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " not found in patch mesh", c);
    PetscCall(ISRestoreIndices(subpIS, &points));
  }
  PetscCall(DMGetLocalSection(patch, &s));
  PetscCall(DMGetGlobalSection(patch, &gs));
  PetscCall(DMGetLocalSection(rpatch, &sRef));
  PetscCall(DMGetGlobalSection(rpatch, &gsRef));

  PetscCall(PetscSectionGetChart(gsRef, &pStart, &pEnd));
  for (PetscInt p = pStart, dof, cdof; p < pEnd; ++p) {
    PetscCall(PetscSectionGetFieldConstraintDof(sRef, p, 0, &cdof));
    PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
    Nfine += !cdof && dof > 0 ? dof : 0;
  }
  PetscCall(DMPlexGetTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  for (PetscInt cl = 0, dof, cdof; cl < Ncl*2; cl += 2) {
    PetscCall(PetscSectionGetConstraintDof(s, closure[cl], &cdof));
    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    Ncoarse += !cdof && dof > 0 ? dof : 0;
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "c: %d NCoarse: %d Nfine: %d\n", c, Ncoarse, Nfine));
  PetscCall(PetscCalloc3(Nfine, &rows, Ncoarse, &cols, Nfine * Ncoarse, &elemP));
  PetscCall(DMGetGlobalSection(dm, &dgs));
  PetscCall(DMGetGlobalSection(rdm, &dgsRef));
  PetscCall(ISGetIndices(subpIS, &points));
  PetscCall(PetscSectionGetChart(gsRef, &pStart, &pEnd));
  for (PetscInt p = pStart, i = 0; p < pEnd; ++p) {
    DMPolytopeType cct;  // Celltype for parrent point in coarse patch
    DMPolytopeType ct;   // Celltype for point p in refined patch
    PetscInt       cp;   // Parent point in coarse patch
    PetscInt       r;    // Replica number for point p in refined patch
    PetscInt       dcp;  // Parent point in coarse domain
    PetscInt       dp;   // Point p in refined domain
    PetscInt       doff; // Offset in global vector for refined domain
    PetscInt       dof, cdof;

    PetscCall(PetscSectionGetFieldConstraintDof(sRef, p, 0, &cdof));
    PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
    // Get parent point in coarse patch for this point in refined patch
    PetscCall(DMPlexTransformGetSourcePoint(((DM_Plex*)rpatch->data)->tr, p, &cct, &ct, &cp, &r));
    // Convert parent point in coarse patch to point in coarse domain
    dcp = points[cp];
    // Get points produced by domain point
    PetscCall(DMPlexTransformGetTargetPoint(((DM_Plex*)rdm->data)->tr, cct, ct, dcp, r, &dp));
    PetscCall(PetscSectionGetFieldOffset(dgsRef, dp, 0, &doff));
    for (PetscInt d = 0; d < (cdof ? 0 : dof); ++d) rows[i++] = doff + d;
  }
  for (PetscInt cl = 0, i = 0; cl < Ncl*2; cl += 2) {
    PetscInt dof, cdof, doff;

    PetscCall(PetscSectionGetFieldConstraintDof(s, closure[cl], 0, &cdof));
    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    PetscCall(PetscSectionGetFieldOffset(dgs, points[closure[cl]], 0, &doff));
    for (PetscInt d = 0; d < (cdof ? 0 : dof); ++d) cols[i++] = doff + d;
  }
  PetscCall(ISRestoreIndices(subpIS, &points));
  PetscCall(DMPlexRestoreTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  PetscCall(DMCreateInterpolation(patch, rpatch, &pP, NULL));

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)rpatch), &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "patch_"));
  PetscCall(SNESSetDM(snes, rpatch));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMPlexSetSNESLocalFEM(rpatch, ctx, ctx, ctx));

  PetscCall(DMCreateMassMatrix(rpatch, rpatch, &pM));
  PetscCall(DMGetGlobalVector(rpatch, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "potential"));
  PetscCall(DMGetGlobalVector(rpatch, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(DMGetGlobalVector(rpatch, &psi));
  PetscCall(PetscObjectSetName((PetscObject)psi, "fine basis"));
  PetscCall(DMGetGlobalVector(patch, &cpsi));
  PetscCall(PetscObjectSetName((PetscObject)cpsi, "coarse basis"));
  PetscCall(VecZeroEntries(cpsi));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  if (viewPatchSol) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "Patch for cell %" PetscInt_FMT "\n", c));
  {
    Mat J;

    PetscCall(SNESSetUp(snes));
    PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Patch_%" PetscInt_FMT "_Jacobian", c));
    PetscCall(PetscObjectSetName((PetscObject)J, name));
  }
  PetscCall(DMPlexGetTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  for (PetscInt cl = 0; cl < Ncl*2; cl += 2) {
    const PetscScalar *a;
    PetscInt           dof, cdof, off;

    PetscCall(PetscSectionGetConstraintDof(s, closure[cl], &cdof));
    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    PetscCall(PetscSectionGetFieldOffset(gs, closure[cl], 0, &off));
    for (PetscInt d = 0; d < (cdof ? 0 : dof); ++d, ++j) {
      PetscInt foff = 0;
      {
        Vec rhs;

        PetscCall(SNESGetFunction(snes, &rhs, NULL, NULL));
        PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Patch_%" PetscInt_FMT "_Rhs_%" PetscInt_FMT, c, j));
        PetscCall(PetscObjectSetName((PetscObject)rhs, name));
      }
      PetscCall(VecSetValue(cpsi, off+d, 1., INSERT_VALUES));
      PetscCall(MatMult(pP, cpsi, psi));
      // We need the weak rhs, not the coefficients
      PetscCall(MatMult(pM, psi, b));
      PetscCall(SNESSolve(snes, b, u));
      PetscCall(VecSetValue(cpsi, off+d, 0., INSERT_VALUES));
      // Insert values into element matrix
      PetscCall(VecGetArrayRead(u, &a));
      // Copy out first field
      for (PetscInt p = pStart; p < pEnd; ++p) {
        PetscInt dof, cdof, off;

        PetscCall(PetscSectionGetFieldConstraintDof(sRef, p, 0, &cdof));
        PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
        PetscCall(PetscSectionGetFieldOffset(gsRef, p, 0, &off));
        for (PetscInt d = 0; d < (cdof > 0 ? 0 : dof); ++d, ++foff) {
          if (PetscAbsScalar(a[off + d]) > PETSC_SMALL) elemP[(foff + d) * Ncoarse + j] = a[off + d];
        }
      }
      PetscCheck(foff == Nfine, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Size of basis vector %" PetscInt_FMT ": %" PetscInt_FMT " does not match fine space %" PetscInt_FMT " for cell %" PetscInt_FMT, j, foff, Nfine, c);
      PetscCall(VecRestoreArrayRead(u, &a));
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  PetscCheck(j == Ncoarse, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of columns %" PetscInt_FMT " != %" PetscInt_FMT " coarse space size", j, Ncoarse);
  if (viewPatchSol) PetscCall(PrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, P, c, Nfine, rows, Ncoarse, cols, elemP));
  if (viewPatchSol && Ncoarse > 0) {
    PetscViewer binViewer;
    Mat         patchP;
    char        filename[PETSC_MAX_PATH_LEN];

    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &binViewer));
    PetscCall(PetscViewerSetType(binViewer, PETSCVIEWERBINARY));
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "/PETSc3/petsc/petsc-dev/%d_proj_mat.bin", c));
    PetscCall(PetscViewerFileSetName(binViewer, filename));
    PetscCall(PetscViewerFileSetMode(binViewer, FILE_MODE_WRITE));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nfine, Ncoarse, elemP, &patchP));
    PetscCall(MatView(patchP, binViewer));
    PetscCall(MatDestroy(&patchP));
    PetscCall(PetscViewerDestroy(&binViewer));
  }
  PetscCall(MatSetValues(P, Nfine, rows, Ncoarse, cols, elemP, INSERT_VALUES));
  PetscCall(DMRestoreGlobalVector(rpatch, &u));
  PetscCall(DMRestoreGlobalVector(rpatch, &b));
  PetscCall(DMRestoreGlobalVector(rpatch, &psi));
  PetscCall(DMRestoreGlobalVector(patch, &cpsi));
  PetscCall(MatDestroy(&pM));
  PetscCall(MatDestroy(&pP));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFree3(rows, cols, elemP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Make a patch for each cell
//   TODO: Just need a patch covering each dof
PetscErrorCode DMPlexCreatePatchBasis(DM dm, Mat *basis)
{
  DM        rdm;
  Mat       P;
  PetscInt  cStart, cEnd;
  PetscBool useCone, useClosure;

  PetscFunctionBegin;
  PetscCall(DMPlexRefineMesh_Private(dm, &rdm));
  PetscCall(DMCopyDisc(dm, rdm));
  PetscCall(DMPlexCreateProlongator_Private(dm, rdm, &P));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_TRUE));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    DM patch, rpatch;

    PetscCall(DMPlexCreatePatch(dm, c, &patch));
    PetscCall(DMPlexRefineMesh_Private(patch, &rpatch));
    PetscCall(DMPlexPatchSolve(dm, patch, c, rdm, rpatch, P));
    PetscCall(DMDestroy(&rpatch));
    PetscCall(DMDestroy(&patch));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscCall(DMSetBasicAdjacency(dm, useCone, useClosure));

  PetscCall(DMDestroy(&rdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
