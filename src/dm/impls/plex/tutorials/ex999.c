static char help[] = "Tutorial\n\n";

#include <petscsnes.h>
#include <petscdmplex.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscfe.h>
#include <petsc/private/petscfeimpl.h>
#include <petscdmplextransform.h>
#include <petsc/private/sectionimpl.h>  /* Only here for _PetscSFCreateSectionSF */

#define CELL_LABEL_E 101
#define CELL_LABEL_A 102
#define INTERFACE_LABEL_VALUE 104

typedef struct {
  PetscInt  dim;     /* The topological dimension */
} AppCtx;

typedef void (*ComputeJcobianType)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);

typedef PetscErrorCode (*ProjectFunctionType)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*);

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  options->dim     = 3;

  ierr = PetscOptionsBegin(comm, "", "FE Integration Performance Options", "PETSCFE");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Some submesh functions */

static PetscErrorCode DMLabelCreateByClosure(DM dm, DMLabel label, PetscInt labelValue, const char labelClosureName[], DMLabel *labelClosure)
{
  PetscInt        labelSize, p;
  IS              labelIS;
  const PetscInt *points;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMLabelCreate(PetscObjectComm((PetscObject)label), labelClosureName, labelClosure);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(label, labelValue, &labelSize);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, labelValue, &labelIS);CHKERRQ(ierr);
  if (labelIS) {ierr = ISGetIndices(labelIS, &points);CHKERRQ(ierr);}
  for (p = 0; p < labelSize; ++p) {
    PetscInt  point = points[p], clSize, cl, *closure = NULL;

    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < clSize*2; cl += 2) {ierr = DMLabelSetValue(*labelClosure, closure[cl], labelValue);CHKERRQ(ierr);}
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
  }
  if (labelIS) {ierr = ISRestoreIndices(labelIS, &points);CHKERRQ(ierr);}
  ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateLabelWithNoGhostPoints(DM dm, DMLabel inFilter, const char name[], DMLabel *outFilter)
{
  PetscSF         pointSF;
  PetscInt        nleaves, i, p;
  const PetscInt *ilocal = NULL;
  DMLabel         ghostLabel;
  IS              inFilterIS;
  PetscInt        inFilterSize;
  const PetscInt *inFilterPoints = NULL;
  PetscBool       has;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMLabelCreate(PetscObjectComm((PetscObject)dm), name, outFilter);
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF, NULL, &nleaves, &ilocal, NULL);CHKERRQ(ierr);
  ierr = DMLabelCreate(PetscObjectComm((PetscObject)dm), "ghost", &ghostLabel);
  for (i = 0; i < nleaves; ++i) {
    p = ilocal ? ilocal[i] : i;
    ierr = DMLabelSetValue(ghostLabel, p, 1);CHKERRQ(ierr);
  }
  ierr = DMLabelGetStratumSize(inFilter, INTERFACE_LABEL_VALUE, &inFilterSize);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(inFilter, INTERFACE_LABEL_VALUE, &inFilterIS);CHKERRQ(ierr); 
  if (inFilterIS) {ierr = ISGetIndices(inFilterIS, &inFilterPoints);CHKERRQ(ierr);}
  for (i = 0; i < inFilterSize; ++i) {
    p = inFilterPoints[i];
    ierr = DMLabelStratumHasPoint(ghostLabel, 1, p, &has);CHKERRQ(ierr);
    if (!has) {ierr = DMLabelSetValue(*outFilter, p, INTERFACE_LABEL_VALUE);CHKERRQ(ierr);}
  }
  if (inFilterIS) {ierr = ISRestoreIndices(inFilterIS, &inFilterPoints);CHKERRQ(ierr);}
  ierr = ISDestroy(&inFilterIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&ghostLabel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSetGlobalSubpointMap(DM dm, DM subdm)
{
  PetscInt        dim, d;
  IS              globalPointNumbers;
  const PetscInt *gpoints = NULL;
  DMLabel         subpointMap, globalSubpointMap;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(subdm, "globalSubpointMap");CHKERRQ(ierr);
  ierr = DMGetLabel(subdm, "globalSubpointMap", &globalSubpointMap);CHKERRQ(ierr);
  ierr = DMGetDimension(subdm, &dim);CHKERRQ(ierr);
  ierr = DMPlexCreatePointNumbering(dm, &globalPointNumbers);CHKERRQ(ierr);
  if (globalPointNumbers) {ierr = ISGetIndices(globalPointNumbers, &gpoints);CHKERRQ(ierr);}
  ierr = DMPlexGetSubpointMap(subdm, &subpointMap);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    IS              subpointIS;
    const PetscInt *subpoints = NULL;
    PetscInt        n, i, gpoint;

    ierr = DMLabelGetStratumSize(subpointMap, d, &n);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(subpointMap, d, &subpointIS);CHKERRQ(ierr);
    if (subpointIS) {ierr = ISGetIndices(subpointIS, &subpoints);CHKERRQ(ierr);}
    for (i = 0; i < n; ++i) {
      gpoint = gpoints[subpoints[i]];
      if (gpoint < 0) {gpoint = -1 - gpoint;}
      ierr = DMLabelSetValue(globalSubpointMap, gpoint, d);CHKERRQ(ierr);
    }
    if (subpointIS) {ierr = ISRestoreIndices(subpointIS, &subpoints);CHKERRQ(ierr);}
    ierr = ISDestroy(&subpointIS);CHKERRQ(ierr);
  }
  if (globalPointNumbers) {ierr = ISRestoreIndices(globalPointNumbers, &gpoints);CHKERRQ(ierr);}
  ierr = ISDestroy(&globalPointNumbers);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSubmeshCreateFilteredSubpoints(DM subdm, DMLabel filter, PetscInt filterValue, PetscInt *numPoints, PetscInt **points, PetscInt **subpoints)
{
  PetscInt        subdim, d, n = 0;
  DMLabel         subpointMap;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMLabelGetStratumSize(filter, filterValue, numPoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(*numPoints, subpoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(*numPoints, points);CHKERRQ(ierr);
  ierr = DMGetDimension(subdm, &subdim);CHKERRQ(ierr);
  ierr = DMGetLabel(subdm, "globalSubpointMap", &subpointMap);CHKERRQ(ierr);
  for (d = 0; d <= subdim; ++d) {
    PetscInt        pStart, pEnd, p;
    IS              stratumIS;
    const PetscInt *stratumPoints;
    PetscBool       has;

    ierr = DMPlexGetDepthStratum(subdm, d, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(subpointMap, d, &stratumIS);CHKERRQ(ierr);
    if (stratumIS) {ierr = ISGetIndices(stratumIS, &stratumPoints);CHKERRQ(ierr);}
    for (p = pStart; p < pEnd; ++p) {
      ierr = DMLabelStratumHasPoint(filter, filterValue, p, &has);CHKERRQ(ierr);
      if (has) {
        (*subpoints)[n] = p;
        (*points)[n] = stratumPoints[p - pStart];
        n++;
      }
    }
    if (stratumIS) {ierr = ISRestoreIndices(stratumIS, &stratumPoints);CHKERRQ(ierr);}
    ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
  }
  if (n != *numPoints) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Mismatching number of indices: %D != %D", n, *numPoints);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSubmeshCreateInterSubmeshPointSF(DM rootSubdm, DM leafSubdm, DMLabel rootFilter, PetscInt rootFilterValue, DMLabel leafFilter, PetscInt leafFilterValue, PetscSF *sf)
{
  PetscInt        numRootPoints, *rootPoints, *rootSubpoints, numLeafPoints, *leafPoints, *leafSubpoints, p, minPoint = PETSC_MAX_INT, maxPoint = PETSC_MIN_INT;
  PetscLayout     layout;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexSubmeshCreateFilteredSubpoints(rootSubdm, rootFilter, rootFilterValue, &numRootPoints, &rootPoints, &rootSubpoints);CHKERRQ(ierr);
  ierr = DMPlexSubmeshCreateFilteredSubpoints(leafSubdm, leafFilter, leafFilterValue, &numLeafPoints, &leafPoints, &leafSubpoints);CHKERRQ(ierr);
  /* Shift parent points so that they reside in [0, maxPoint - minPoint) */
  for (p = 0; p < numRootPoints; ++p) {
    minPoint = PetscMin(minPoint, rootPoints[p]);
    maxPoint = PetscMax(maxPoint, rootPoints[p]);
  }
  maxPoint += 1;
  ierr = MPI_Allreduce(MPI_IN_PLACE, &minPoint, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)rootSubdm));CHKERRMPI(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &maxPoint, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)rootSubdm));CHKERRMPI(ierr);
  for (p = 0; p < numRootPoints; ++p) {
    rootPoints[p] -= minPoint;
    leafPoints[p] -= minPoint;
  }
  /* Create layout representing [0, maxPoint - minPoint) */
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)rootSubdm), &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(layout, maxPoint - minPoint);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscSFCreateByMatchingIndices(layout, numRootPoints, rootPoints, rootSubpoints, 0, numLeafPoints, leafPoints, leafSubpoints, 0, NULL, sf);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscFree(rootPoints);CHKERRQ(ierr);
  ierr = PetscFree(rootSubpoints);CHKERRQ(ierr);
  ierr = PetscFree(leafPoints);CHKERRQ(ierr);
  ierr = PetscFree(leafSubpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateFiniteElementMatrix(DM dmorig, PetscInt numComp, ComputeJcobianType K00, ComputeJcobianType K01, ComputeJcobianType K10, ComputeJcobianType K11, const char feprefix[], AppCtx *user, Mat *M)
{
  DM              dm;
  PetscDS         ds;
  PetscFE         fe;
  SNES            snes;
  Vec             U;
  PetscInt        dim;
  PetscBool       simplex;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMClone(dmorig, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, numComp, simplex, feprefix, -1, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, K00, K01, K10, K11);CHKERRQ(ierr);
  ierr = SNESCreate(PetscObjectComm((PetscObject)dm), &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, user, user, user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, M);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &U);CHKERRQ(ierr); /* place holder */
  ierr = SNESComputeJacobian(snes, U, *M, *M);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Fix PetscSFCreateSectionSF() to allow for remoteOffsets = NULL */

static PetscErrorCode _PetscSFCreateSectionSF(PetscSF sf, PetscSection rootSection, PetscInt remoteOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  PetscInt       *_remoteOffsets = NULL;
  PetscBool       flag, gflag;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /*TODO: in debug mode check rootSection and leafSection are consistent under sf */
  flag = (PetscBool)(!remoteOffsets);
  ierr = MPIU_Allreduce(&flag, &gflag, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)sf));CHKERRMPI(ierr);
  if (gflag) {
    PetscInt  rpStart, rpEnd, lpStart, lpEnd;

    ierr = PetscSectionGetChart(rootSection, &rpStart, &rpEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(leafSection, &lpStart, &lpEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(lpEnd - lpStart, &_remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf, MPIU_INT, &rootSection->atlasOff[-rpStart], &_remoteOffsets[-lpStart], MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, &rootSection->atlasOff[-rpStart], &_remoteOffsets[-lpStart], MPI_REPLACE);CHKERRQ(ierr);
  } else {
    _remoteOffsets = remoteOffsets;
  }
  ierr = PetscSFCreateSectionSF(sf, rootSection, _remoteOffsets, leafSection, sectionSF);CHKERRQ(ierr);
  if (gflag) {ierr = PetscFree(_remoteOffsets);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSubmeshCreateInterSubmeshDofSF(DM rootDM, DM leafDM, PetscSection rootSection, PetscSection leafSection, DMLabel rootFilter, PetscInt rootFilterValue, DMLabel leafFilter, PetscInt leafFilterValue, PetscSF *dofSF)
{
  PetscSF         pointSF;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexSubmeshCreateInterSubmeshPointSF(rootDM, leafDM, rootFilter, rootFilterValue, leafFilter, leafFilterValue, &pointSF);CHKERRQ(ierr);
  ierr = _PetscSFCreateSectionSF(pointSF, rootSection, NULL, leafSection, dofSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSubmeshInheritLabel(DM dm, DM subdm, const char *labelname, const char *sublabelname, PetscInt height)
{
  DMLabel         subpointMap, interfaceLabel, parentInterfaceLabel;
  IS              subpointIS;
  const PetscInt *subpoints;
  PetscInt        pStart, pEnd, p, subdim;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(subdm, sublabelname);CHKERRQ(ierr);
  ierr = DMGetLabel(subdm, sublabelname, &interfaceLabel);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, labelname, &parentInterfaceLabel);CHKERRQ(ierr);

  ierr = DMPlexGetSubpointMap(subdm, &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(subdm, height, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMGetDimension(subdm, &subdim);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(subpointMap, subdim-height, &subpointIS);CHKERRQ(ierr);
  if (subpointIS) {ierr = ISGetIndices(subpointIS, &subpoints);CHKERRQ(ierr);}
  for (p = pStart; p < pEnd; ++p) {
    PetscInt v;

    ierr = DMLabelGetValue(parentInterfaceLabel, subpoints[p-pStart], &v);CHKERRQ(ierr);
    if (v > 0) {ierr = DMLabelSetValue(interfaceLabel, p, v);CHKERRQ(ierr);}
  }
  if (subpointIS) {ierr = ISRestoreIndices(subpointIS, &subpoints);CHKERRQ(ierr);}
  ierr = ISDestroy(&subpointIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Mesh */

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscInt         dim = 3, ncells = 2, nverts = 5, ncellverts = 4;
  const PetscInt   cells[] = {0, 1, 2, 3,
                              1, 2, 3, 4};
  const PetscReal  coords[] = {0, 0, 0,
                               0, 0, 1,
                               0, 1, 0,
                               1, 0, 0,
                               1, 1, 1};
  DMLabel          cellLabel, faceLabel;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateFromCellListPetsc(comm, dim, ncells, nverts, ncellverts, PETSC_TRUE, cells, dim, coords, dm);CHKERRQ(ierr);
  ierr = DMCreateLabel(*dm, "Cell Sets");CHKERRQ(ierr);
  ierr = DMGetLabel(*dm, "Cell Sets", &cellLabel);CHKERRQ(ierr);
  ierr = DMLabelSetValue(cellLabel, 0, CELL_LABEL_E);CHKERRQ(ierr);
  ierr = DMLabelSetValue(cellLabel, 1, CELL_LABEL_A);CHKERRQ(ierr);
  ierr = DMCreateLabel(*dm, "Face Sets");CHKERRQ(ierr);
  ierr = DMGetLabel(*dm, "Face Sets", &faceLabel);CHKERRQ(ierr);
  ierr = DMLabelSetValue(faceLabel, 10, INTERFACE_LABEL_VALUE);CHKERRQ(ierr);
  {
    DM        pdm;
    PetscInt  overlap = 1;
    PetscSF   sf;

    ierr = DMPlexDistribute(*dm, overlap, &sf, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = pdm;
    }
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Elastodynamics */

static PetscErrorCode init_V(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1*x[0] + 10* x[1] + 100*x[2]+2;
  u[1] = 5*x[0] + 6*x[1] + 7*x[2]+1;
  u[2] = 1*x[0] + 77*x[1] + 999*x[2]+3;
  return 0;
}

static PetscErrorCode init_velocity_local(DM dm, Vec V)
{
  ProjectFunctionType  funcs[] = {init_V};
  PetscErrorCode       ierr;

  ierr = VecSet(V, 0.0);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, V);CHKERRQ(ierr);
  return 0;
}

static void mass_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt i, j;

  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      f1[dim * i + j] = (i == j) ? 1.0 : 0.0;
    }
  }
}

static void ResidualU1(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f1[])
{
  PetscInt i, j;

  for (i=0; i<dim; ++i) {
    for (j=0; j<dim; ++j) f1[dim*i + j] = u_x[dim*i + j];
  }
}

static void JacobianU1U1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt i0, j0, i1, j1;

  for (i0=0; i0<dim; ++i0) {
    for (i1=0; i1<dim; ++i1) {
      for (j0=0; j0<dim; ++j0) {
        for (j1=0; j1<dim; ++j1) {
          f1[dim*dim*dim*i0 + dim*dim*i1 + dim*j0 + j1] = (i0 == i1 && j0 == j1) ? 1.0 : 0.0;
        }
      }
    }
  }
}

static void interfaceConditionMantle(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt  d;

  for (d = 0; d < dim; ++d) f0[0] = a[0] * n[d];
}

static PetscErrorCode SetupPrimalProblemE(DM dm, AppCtx *user)
{
  PetscDS         ds;
  PetscWeakForm   wf;
  DMLabel         label;
  const PetscInt  id = INTERFACE_LABEL_VALUE;
  PetscInt        bcid;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 0, NULL, ResidualU1);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, JacobianU1U1);CHKERRQ(ierr);
  //ierr = PetscDSSetExactSolution(ds, 0, trig_u, user);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Face Sets", &label);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(ds, DM_BC_NATURAL, "p_to_u", label, 1, &id, 0, 0, NULL, (void (*)(void))NULL, NULL, user, &bcid);CHKERRQ(ierr);
  /* ierr = PetscDSSetBdResidual(ds, 0, interfaceCondition0, NULL);CHKERRQ(ierr); ?*/
  ierr = PetscDSGetBoundary(ds, bcid, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, interfaceConditionMantle, 0, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretizationE(DM dm, AppCtx *user)
{
  PetscFE        fe;
  PetscInt       dim, numComp=3;
  PetscBool      simplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  /* Set up primary field */
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, numComp, simplex, "displacement_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fe, "displacement");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  /* Set up auxiliary field */
  {
    DM             dmAux, coordDM;
    PetscFE        feAuxp;
    Vec            adotn;

    ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, "pressure_", -1, &feAuxp);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)feAuxp, "adotn");CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe, feAuxp);CHKERRQ(ierr);
    ierr = DMSetField(dmAux, 0, NULL, (PetscObject)feAuxp);CHKERRQ(ierr);
    ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dmAux, &adotn);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)adotn, "adotn");CHKERRQ(ierr);
    ierr = DMSetAuxiliaryVec(dm, NULL, 0, adotn);CHKERRQ(ierr);
    ierr = VecDestroy(&adotn);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&feAuxp);CHKERRQ(ierr);
    ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /*  */
  ierr = SetupPrimalProblemE(dm, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Acoustics */

static PetscErrorCode init_P(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 9*x[0] + 90* x[1] + 999*x[2]+2;
  return 0;
}

static void interfaceCondition0(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt  d;
printf("printintg a: %f\n", a[0]);
  for (d=0; d<dim; ++d) f0[0] = x[d]*n[d];
}

static void mass_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  *f1 = 1.0;
}

static PetscErrorCode SetupPrimalProblemA(DM dm, AppCtx *user)
{
  PetscDS         ds;
  PetscWeakForm   wf;
  DMLabel         label;
  const PetscInt  id = INTERFACE_LABEL_VALUE;
  PetscInt        bcid;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  //ierr = PetscDSSetResidual(ds, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
  //ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  //ierr = PetscDSSetExactSolution(ds, 0, trig_u, user);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Face Sets", &label);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(ds, DM_BC_NATURAL, "p_to_u", label, 1, &id, 0, 0, NULL, (void (*)(void))NULL, NULL, user, &bcid);CHKERRQ(ierr);
  /* ierr = PetscDSSetBdResidual(ds, 0, interfaceCondition0, NULL);CHKERRQ(ierr); ?*/
  ierr = PetscDSGetBoundary(ds, bcid, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, 0, interfaceCondition0, 0, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretizationA(DM dm, AppCtx *user)
{
  PetscFE        fe;
  PetscInt       dim, numComp = 1;
  PetscBool      simplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, numComp, simplex, "pressure_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fe, "pressure");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupPrimalProblemA(dm, user);CHKERRQ(ierr);
  {
    DM             dmAux, coordDM;
    PetscFE        feAux;
    Vec            auxVec;

    ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 3, simplex, "displacement_", -1, &feAux);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe, feAux);CHKERRQ(ierr);
    ierr = DMSetField(dmAux, 0, NULL, (PetscObject)feAux);CHKERRQ(ierr);
    ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dmAux, &auxVec);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)auxVec, "auxVecforpressure");CHKERRQ(ierr);
    ierr = DMSetAuxiliaryVec(dm, NULL, 0, auxVec);CHKERRQ(ierr);
    ierr = VecDestroy(&auxVec);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&feAux);CHKERRQ(ierr);
    ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _update_auxiliary_vec(DM dm, Vec source, PetscSF sf)
{
  Vec                target;
  PetscScalar       *targetArray;
  const PetscScalar *sourceArray;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &target);CHKERRQ(ierr);
  ierr = VecGetArrayRead(source, &sourceArray);CHKERRQ(ierr);
  ierr = VecGetArray(target, &targetArray);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_SCALAR, sourceArray, targetArray, MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_SCALAR, sourceArray, targetArray, MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArray(target, &targetArray);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(source, &sourceArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ElastoAcousticRK4Update(DM dmE, DM dmA, Vec u, Vec v, Vec p, Vec q, KSP kspE, KSP kspA, Mat Kee, SNES snesEA, SNES snesAE, Mat Kaa, PetscSF sfEA, PetscSF sfAE, Vec udot, Vec vdot, Vec pdot, Vec qdot)
{
  Vec            f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /*  */
  ierr = _update_auxiliary_vec(dmE, q, sfAE);CHKERRQ(ierr);
  ierr = _update_auxiliary_vec(dmA, v, sfEA);CHKERRQ(ierr);
  /* udot = v  */
  ierr = VecCopy(v, udot);CHKERRQ(ierr);
  /* vdot = Kee * u + Kae * q */
  ierr = DMGetGlobalVector(dmE, &f);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snesEA, u, f);CHKERRQ(ierr);
  ierr = MatMultAdd(Kee, u, f, vdot);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmE, &f);CHKERRQ(ierr);
  ierr = KSPSolve(kspE, vdot, vdot);CHKERRQ(ierr);
  /* pdot = q */
  ierr = VecCopy(q, pdot);CHKERRQ(ierr);
  /* qdot = Kea * v + Kaa * p */
  ierr = DMGetGlobalVector(dmA, &f);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snesAE, p, f);CHKERRQ(ierr);
  ierr = MatMultAdd(Kee, p, f, qdot);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmA, &f);CHKERRQ(ierr);
  ierr = KSPSolve(kspA, qdot, qdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm, dmE, dmA;
  DM              auxdmE, auxdmA;
  Vec             auxVecE, auxVecA;
  PetscInt        dim;
  PetscBool       simplex;
  DMLabel         cellLabel;
  PetscSF         dofSFEA, dofSFAE;
  AppCtx          user;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Cell Sets", &cellLabel);CHKERRQ(ierr);
  /* Create dmE */
  ierr = DMPlexFilter(dm, cellLabel, CELL_LABEL_E, &dmE);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dmE, "dmE");CHKERRQ(ierr);
  ierr = DMPlexSubmeshInheritLabel(dm, dmE, "Face Sets", "Face Sets", 1);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmE, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = SetupDiscretizationE(dmE, &user);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dmE, NULL, 0, &auxVecE);CHKERRQ(ierr);
  ierr = VecGetDM(auxVecE, &auxdmE);CHKERRQ(ierr);
  ierr = DMPlexSetGlobalSubpointMap(dm, dmE);CHKERRQ(ierr);
  /* Create dmA */
  ierr = DMPlexFilter(dm, cellLabel, CELL_LABEL_A, &dmA);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dmA, "dmA");CHKERRQ(ierr);
  ierr = DMPlexSubmeshInheritLabel(dm, dmA, "Face Sets", "Face Sets", 1);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmA, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = SetupDiscretizationA(dmA, &user);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dmA, NULL, 0, &auxVecA);CHKERRQ(ierr);
  ierr = VecGetDM(auxVecA, &auxdmA);CHKERRQ(ierr);
  ierr = DMPlexSetGlobalSubpointMap(dm, dmA);CHKERRQ(ierr);
  /* Create DOF SFs: root global Vec -> leaf local Vec */
  {
    PetscSF       pointSFE, pointSFA;
    DMLabel       labelE, labelA, filterE, filterE1, filterA, filterA1;
    PetscSection  sectionE, auxSectionE, sectionA, auxSectionA, gsectionE, gsectionA;

    ierr = DMGetLabel(dmE, "Face Sets", &labelE);CHKERRQ(ierr);
    ierr = DMGetLabel(dmA, "Face Sets", &labelA);CHKERRQ(ierr);
    ierr = DMLabelCreateByClosure(dmE, labelE, INTERFACE_LABEL_VALUE, "filterE", &filterE);CHKERRQ(ierr);
    ierr = DMLabelCreateByClosure(dmA, labelA, INTERFACE_LABEL_VALUE, "filterA", &filterA);CHKERRQ(ierr);
    /* DOF SF to move u_tt from dmE to dmA */
    ierr = DMPlexCreateLabelWithNoGhostPoints(dmE, filterE, "filterE1", &filterE1);
    ierr = DMGetLocalSection(dmE, &sectionE);CHKERRQ(ierr);
    ierr = DMGetPointSF(dmE, &pointSFE);CHKERRQ(ierr);
    ierr = PetscSectionCreateGlobalSection(sectionE, pointSFE, PETSC_FALSE, PETSC_TRUE, &gsectionE);CHKERRQ(ierr);
    ierr = DMGetLocalSection(auxdmA, &auxSectionA);CHKERRQ(ierr);
    ierr = DMPlexSubmeshCreateInterSubmeshDofSF(dmE, dmA, gsectionE, auxSectionA, filterE1, INTERFACE_LABEL_VALUE, filterA, INTERFACE_LABEL_VALUE, &dofSFEA);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&filterE1);CHKERRQ(ierr);
    /* DOF SF to move p from dmA to dmE */
    ierr = DMPlexCreateLabelWithNoGhostPoints(dmA, filterA, "filterA1", &filterA1);
    ierr = DMGetLocalSection(dmA, &sectionA);CHKERRQ(ierr);
    ierr = DMGetPointSF(dmA, &pointSFA);CHKERRQ(ierr);
    ierr = PetscSectionCreateGlobalSection(sectionA, pointSFA, PETSC_FALSE, PETSC_TRUE, &gsectionA);CHKERRQ(ierr);
    ierr = DMGetLocalSection(auxdmE, &auxSectionE);CHKERRQ(ierr);
    ierr = DMPlexSubmeshCreateInterSubmeshDofSF(dmA, dmE, gsectionA, auxSectionE, filterA1, INTERFACE_LABEL_VALUE, filterE, INTERFACE_LABEL_VALUE, &dofSFAE);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&filterA1);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&filterE);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&filterA);CHKERRQ(ierr);
  }
  /* RK4 */
  {
  SNES            snesE, snesA;
    Vec                VE, auxVE, VA, auxVA;
    const PetscScalar *VEArray, *VAArray;
    PetscScalar       *auxVEArray, *auxVAArray;
Mat  A, MassE, MassA;
Vec  F, UU;
ierr = DMCreateMatrix(dmE, &A);CHKERRQ(ierr);
DMCreateGlobalVector(dmE, &F);
DMCreateGlobalVector(dmE, &UU);

  ierr = CreateFiniteElementMatrix(dmE, 3, mass_uu, NULL, NULL, NULL, "displacement_", &user, &MassE);CHKERRQ(ierr);


ierr = MatDestroy(&MassE);CHKERRQ(ierr);
  /* Create snesE */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snesE);CHKERRQ(ierr);
  ierr = SNESSetDM(snesE,dmE);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dmE,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snesE);CHKERRQ(ierr);
//  ierr = DMSNESCheckFromOptions(snesE,U);CHKERRQ(ierr);

    ierr = DMCreateLocalVector(dmE, &VE);CHKERRQ(ierr);
    ierr = init_velocity_local(dmE, VE);CHKERRQ(ierr);
ierr = SNESComputeFunction(snesE, UU, F);CHKERRQ(ierr);
//VecView(F, NULL);
VecSet(F, 0);
ierr = SNESComputeJacobian(snesE, UU, A, A);CHKERRQ(ierr);
ierr = MatMult(A, UU, F);CHKERRQ(ierr);
VecView(F, NULL);
//MatView(A, NULL);
VecDestroy(&F);
VecDestroy(&UU);
    ierr = DMGetAuxiliaryVec(dmA, NULL, 0, &auxVA);CHKERRQ(ierr);
    ierr = VecSet(auxVA, 0.0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(VE, &VEArray);CHKERRQ(ierr);
    ierr = VecGetArray(auxVA, &auxVAArray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(dofSFEA, MPIU_SCALAR, VEArray, auxVAArray, MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(dofSFEA, MPIU_SCALAR, VEArray, auxVAArray, MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(VE, &VEArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(auxVA, &auxVAArray);CHKERRQ(ierr);
    ierr = VecDestroy(&VE);CHKERRQ(ierr);
    ierr = init_velocity_local(auxdmA, auxVA);CHKERRQ(ierr);
VecView(auxVA, NULL);
    ierr = SNESDestroy(&snesE);CHKERRQ(ierr);
  }
  /* RK4 */
  {
    Mat  MassE, MassA;
    Vec  F, UU;

    ierr = CreateFiniteElementMatrix(dmE, 3, mass_uu, NULL, NULL, NULL, "displacement_", &user, &MassE);CHKERRQ(ierr);
    ierr = CreateFiniteElementMatrix(dmA, 1, mass_pp, NULL, NULL, NULL, "pressure_", &user, &MassA);CHKERRQ(ierr);

PetscScalar value;
    DMCreateGlobalVector(dmA, &F);
VecSet(F, 1.0);
    DMCreateGlobalVector(dmA, &UU);
    MatMult(MassA, F, UU);
    VecDot(UU, F, &value);
printf("value is %f\n", value);
    VecDestroy(&F);
    VecDestroy(&UU);

    //ierr = ElastoAcousticRK4Update(dmE, dmA, u, v, p, q, kspE, kspA, Kee, snesEA, snesAE, Kaa, dofSFEA, dofSFAE, udot, vdot, pdot, qdot);CHKERRQ(ierr);


    ierr = MatDestroy(&MassE);CHKERRQ(ierr);
    ierr = MatDestroy(&MassA);CHKERRQ(ierr);
  }
  /* Finalize */
  ierr = PetscSFDestroy(&dofSFEA);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&dofSFAE);CHKERRQ(ierr);
  ierr = DMDestroy(&dmE);CHKERRQ(ierr);
  ierr = DMDestroy(&dmA);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    args: -dm_view ascii::ascii_info_detail
    args: -displacement_petscdualspace_type lagrange -displacement_petscspace_degree 2 -displacement_petscspace_components 3
    args: -pressure_petscdualspace_type lagrange -pressure_petscspace_degree 2
TEST*/
