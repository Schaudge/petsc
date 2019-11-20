
#include <petscdmplex.h>
#include <petsc/private/petscimpl.h>
#include <petscbt.h>

typedef PetscInt PetscPolytope;
const PetscPolytope PETSCPOLYTOPE_NONE = -1;

typedef struct _n_PetscPolytopeData *PetscPolytopeData;
typedef struct _n_PetscPolytopeName PetscPolytopeName;

typedef struct _n_PetscPolytopeCone
{
  PetscInt index;
  PetscInt orientation;
} PetscPolytopeCone;

typedef struct _n_PetscPolytopeSupp
{
  PetscInt index;
  PetscInt coneNumber;
} PetscPolytopeSupp;

struct _n_PetscPolytopeData
{
  PetscInt           dim, numFacets, numVertices, numRidges;
  PetscPolytope      *facets;
  PetscBool          *facetsInward;
  PetscInt           *vertexOffsets;
  PetscInt           *facetsToVertices;
  PetscInt           *facetsToVerticesSorted;
  PetscInt           *facetsToVerticesOrder;
  PetscInt           *ridgeOffsets;
  PetscPolytopeCone  *facetsToRidges;
  PetscPolytopeSupp  *ridgesToFacets;
  PetscInt            orientStart, orientEnd, maxFacetSymmetry;
  PetscInt           *orientsToVertexOrders;
  PetscPolytopeCone  *orientsToFacetOrders;
  PetscInt           *orientInverses;
  PetscInt           *orbitProjectors;
  PetscInt           *facetOrientations;
};

static PetscErrorCode PetscPolytopeDataDestroy(PetscPolytopeData *pdata)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pdata) PetscFunctionReturn(0);
  ierr = PetscFree((*pdata)->facets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsInward);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->vertexOffsets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToVertices);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToVerticesSorted);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToVerticesOrder);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->ridgeOffsets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToRidges);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->ridgesToFacets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orientsToVertexOrders);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orientsToFacetOrders);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orientInverses);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orbitProjectors);CHKERRQ(ierr);
  ierr = PetscFree(*pdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataCreate(PetscInt dim, PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscPolytopeData *pData)
{
  PetscPolytopeData pd;
  PetscInt          i, j;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&pd);CHKERRQ(ierr);
  pd->dim         = dim;
  pd->numFacets   = numFacets;
  pd->numVertices = numVertices;
  ierr = PetscMalloc1(numFacets, &(pd->facets));CHKERRQ(ierr);
  ierr = PetscArraycpy(pd->facets, facets, numFacets);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets + 1, &(pd->vertexOffsets));CHKERRQ(ierr);
  ierr = PetscArraycpy(pd->vertexOffsets, vertexOffsets, numFacets + 1);CHKERRQ(ierr);
  ierr = PetscMalloc1(vertexOffsets[numFacets], &(pd->facetsToVertices));CHKERRQ(ierr);
  ierr = PetscArraycpy(pd->facetsToVertices, facetsToVertices, vertexOffsets[numFacets]);CHKERRQ(ierr);
  ierr = PetscMalloc1(vertexOffsets[numFacets], &(pd->facetsToVerticesSorted));CHKERRQ(ierr);
  ierr = PetscMalloc1(vertexOffsets[numFacets], &(pd->facetsToVerticesOrder));CHKERRQ(ierr);
  for (i = 0; i < numFacets; i++) {
    PetscInt *v = &pd->facetsToVertices[pd->vertexOffsets[i]];
    PetscInt *f = &pd->facetsToVerticesSorted[pd->vertexOffsets[i]];
    PetscInt *o = &pd->facetsToVerticesOrder[pd->vertexOffsets[i]];
    PetscInt  n = pd->vertexOffsets[i+1]-pd->vertexOffsets[i];

    for (j = 0; j < n; j++) {
      f[i] = v[i];
      o[i] = i;
    }
    ierr = PetscSortIntWithArray(n, f, o);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(numFacets, &(pd->facetsInward));CHKERRQ(ierr);
  if (numFacets) pd->facetsInward[0] = firstFacetInward;
  *pData = pd;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataCompare(PetscPolytopeData tdata, PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscBool *same)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (numFacets != tdata->numFacets || numVertices != tdata->numVertices || (numFacets > 0 && (firstFacetInward != tdata->facetsInward[0]))) {
    *same = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  else {
    ierr = PetscArraycmp(facets, tdata->facets, numFacets, same);CHKERRQ(ierr);
    if (!*same) PetscFunctionReturn(0);
    ierr = PetscArraycmp(vertexOffsets, tdata->vertexOffsets, numFacets+1, same);CHKERRQ(ierr);
    if (!*same) PetscFunctionReturn(0);
    ierr = PetscArraycmp(facetsToVertices, tdata->facetsToVertices, vertexOffsets[numFacets], same);CHKERRQ(ierr);
    if (!*same) PetscFunctionReturn(0);
    *same = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode PetscPolytopeDataFacetsFromOrientation(PetscPolytopeData data, PetscInt orientation, PetscPolytopeCone facets[])
{
  PetscInt i, o, numFacets;

  PetscFunctionBegin;
  if (orientation < data->orientStart || orientation >= data->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", orientation, data->orientStart, data->orientEnd);
  o = orientation - data->orientStart;
  numFacets = data->numFacets;
  for (i = 0; i < numFacets; i++) facets[i] = data->orientsToFacetOrders[o*numFacets + i];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataOrientationInverse(PetscPolytopeData data, PetscInt orientation, PetscInt *inverse)
{
  PetscFunctionBegin;
  if (orientation < data->orientStart || orientation >= data->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", orientation, data->orientStart, data->orientEnd);
  *inverse = data->orientInverses[orientation - data->orientStart];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataFacetSign(PetscPolytopeData data, PetscInt f, PetscBool *sign)
{
  PetscFunctionBegin;
  if (f < 0 || f >= data->numFacets) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid facet %D (not in [0, %D)", f, data->numFacets);
  *sign = data->facetsInward[f];
  PetscFunctionReturn(0);
}

struct _n_PetscPolytopeName
{
  char          *name;
  PetscPolytope polytope;
};

typedef struct _n_PetscPolytopeSet *PetscPolytopeSet;

struct _n_PetscPolytopeSet
{
  int               numPolytopes;
  int               numPolytopesAlloc;
  PetscPolytopeData *polytopes;
  int               numNames;
  int               numNamesAlloc;
  PetscPolytopeName *names;
};

static PetscPolytopeSet PetscPolytopes = NULL;

static PetscErrorCode PetscPolytopeSetCreate(PetscPolytopeSet *pset)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(pset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetDestroy(PetscPolytopeSet *pset)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pset) PetscFunctionReturn(0);
  PetscValidPointer(pset, 1);
  for (i = 0; i < (*pset)->numPolytopes; i++) {
    ierr = PetscPolytopeDataDestroy(&((*pset)->polytopes[i]));CHKERRQ(ierr);
  }
  ierr = PetscFree((*pset)->polytopes);CHKERRQ(ierr);
  for (i = 0; i < (*pset)->numNames; i++) {
    ierr = PetscFree((*pset)->names[i].name);CHKERRQ(ierr);
  }
  ierr = PetscFree((*pset)->names);CHKERRQ(ierr);
  ierr = PetscFree(*pset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientationCompose(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt a, PetscInt b, PetscInt *aafterb)
{
  PetscPolytopeData p;
  PetscInt f0, o0, f1, o1, oComp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a) {
    *aafterb = b;
    PetscFunctionReturn(0);
  }
  if (!b) {
    *aafterb = a;
    PetscFunctionReturn(0);
  }
  if (polytope < 0 || polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No polytope with id %D\n",polytope);
  p = pset->polytopes[polytope];
  if (a < p->orientStart || a >= p->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", a, p->orientStart, p->orientEnd);
  if (b < p->orientStart || b >= p->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", b, p->orientStart, p->orientEnd);
  f0 = p->orientsToFacetOrders[p->numFacets * (b - p->orientStart)].index;
  o0 = p->orientsToFacetOrders[p->numFacets * (b - p->orientStart)].orientation;
  f1 = p->orientsToFacetOrders[p->numFacets * (a - p->orientStart) + f0].index;
  o1 = p->orientsToFacetOrders[p->numFacets * (a - p->orientStart) + f0].orientation;
  ierr = PetscPolytopeSetOrientationCompose(pset, p->facets[0], o1, o0, &oComp);CHKERRQ(ierr);
  /* find (oComp, f1) */
  *aafterb = p->facetOrientations[f1*p->maxFacetSymmetry + oComp];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientationFromFacet(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt facetOrigin, PetscInt facetImage, PetscInt imageOrientation,
                                                           PetscBool *isOrient, PetscInt *orientation)
{
  PetscPolytopeData p;
  PetscPolytope  f;
  PetscInt       originProj, originSect, imageProj, imageSect;
  PetscInt       oBaseOrigin, oImageBase, oComp;
  PetscInt       baseFacet;
  PetscInt       oBase;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(pset,1);
  if (polytope < 0 || polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No polytope with id %D\n",polytope);
  p = pset->polytopes[polytope];
  if (facetOrigin < 0 || facetOrigin > p->numFacets) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Origin %D not a facet number in [0,%D)\n",facetOrigin,p->numFacets);
  if (facetImage < 0 || facetImage > p->numFacets) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Image %D not a facet number in [0,%D)\n",facetImage,p->numFacets);
  if (p->facets[facetOrigin] != p->facets[facetImage]) { /* not the same facet type, can't be in the same orbit */
    *isOrient = PETSC_FALSE;
    *orientation = PETSC_MIN_INT;
    PetscFunctionReturn(0);
  }
  f = p->facets[facetOrigin];
  originProj = p->orbitProjectors[facetOrigin];
  originSect = p->orientInverses[originProj - p->orientStart];
  imageProj = p->orbitProjectors[facetImage];
  imageSect = p->orientInverses[imageProj - p->orientStart];
  baseFacet = p->orientsToFacetOrders[p->numFacets * (originProj - p->orientStart) + facetOrigin].index;
  if (p->orientsToFacetOrders[p->numFacets * (imageSect - p->orientStart) + baseFacet].index != facetImage) { /* origin and image are not in the same orbit */
    *isOrient = PETSC_FALSE;
    *orientation = PETSC_MIN_INT;
    PetscFunctionReturn(0);
  }
  oBaseOrigin = p->orientsToFacetOrders[p->numFacets * (originSect - p->orientStart) + baseFacet].orientation;
  oImageBase  = p->orientsToFacetOrders[p->numFacets * (imageProj - p->orientStart) + facetImage].orientation;
  ierr = PetscPolytopeSetOrientationCompose(pset, f, imageOrientation, oBaseOrigin, &oComp);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationCompose(pset, f, oImageBase, oComp, &oComp);CHKERRQ(ierr);
  /* imageProj o orientation o originSec = oBase , if orientation exists, maps baseFacet to itself with orientation oComp */
  oBase = p->facetOrientations[baseFacet*p->maxFacetSymmetry + oComp];
  if (oBase < p->orientStart) {
    *isOrient = PETSC_FALSE;
    *orientation = PETSC_MIN_INT;
    PetscFunctionReturn(0);
  }
  /* orientation is imageSec o oBase o originProj */
  ierr = PetscPolytopeSetOrientationCompose(pset, polytope, oBase, originProj, &oComp);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationCompose(pset, polytope, imageSect, oComp, orientation);CHKERRQ(ierr);
  *isOrient = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientationFromVertices(PetscPolytopeSet pset, PetscPolytope polytope, const PetscInt vertices[], PetscBool *isOrientation, PetscInt *orientation)
{
  PetscInt       numVertices;
  PetscPolytopeData data;
  const PetscInt *otov;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (polytope < 0 || polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No polytope with id %D\n",polytope);
  data = pset->polytopes[polytope];
  if (!data->numVertices) {
    *isOrientation = PETSC_TRUE;
    *orientation = 0;
    PetscFunctionReturn(0);
  }
  numVertices = data->numVertices;
  otov        = data->orientsToVertexOrders;
  if (otov) { /* directly compare each vertex order with the order given */
    PetscInt o, oStart, oEnd;

    oStart = data->orientStart;
    oEnd   = data->orientEnd;
    for (o = oStart; o < oEnd; o++) {
      PetscInt v;

      for (v = 0; v < numVertices; v++) if (otov[numVertices * (o - oStart) + v] != vertices[v]) break;
      if (v == numVertices) {
        *isOrientation = PETSC_TRUE;
        *orientation   = o;
      }
    }
    *isOrientation = PETSC_FALSE;
    *orientation   = PETSC_MIN_INT;
  } else {
    PetscInt *vwork, *owork, *order;
    PetscInt i, j, n = data->vertexOffsets[1] - data->vertexOffsets[0];
    PetscInt offset, numFacets = data->numFacets, fo;

    ierr = PetscMalloc2(data->numVertices, &vwork, data->numVertices, &owork);CHKERRQ(ierr);
    /* get the vertices of the first facet */
    for (i = 0; i < n; i++) {
      vwork[i] = vertices[i];
      owork[i] = i;
    }
    /* sort them and permute the order in which they were given */
    ierr = PetscSortIntWithArray(n, vwork, owork);CHKERRQ(ierr);
    for (i = 0; i < numFacets; i++) {
      if (data->facets[i] != data->facets[0]) continue;
      /* compare to the sorted vertices to the sorted vertices for each facet */
      offset = data->vertexOffsets[i];
      for (j = 0; j < n; j++) if (data->facetsToVerticesSorted[j+offset] != vwork[j]) break;
      if (j == n) break;
    }
    if (i == numFacets) { /* vertex order does not correspond to an orientation */
      *isOrientation = PETSC_FALSE;
      *orientation   = PETSC_MIN_INT;
      PetscFunctionReturn(0);
    }
    /* compose the order of the vertices around the image facet and the given order the vertices */
    order = &data->facetsToVerticesOrder[offset];
    for (j = 0; j < n; j++) vwork[owork[j]] = order[j];
    ierr = PetscPolytopeSetOrientationFromVertices(pset, data->facets[0], vwork, isOrientation, &fo);CHKERRQ(ierr);
    ierr = PetscFree2(vwork, owork);CHKERRQ(ierr);
    if (*isOrientation) {
      ierr = PetscPolytopeSetOrientationFromFacet(pset, polytope, 0, i, fo, isOrientation, orientation);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetGetPolytope(PetscPolytopeSet pset, const char name[], PetscPolytope *polytope)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i = 0; i < pset->numNames; i++) {
    PetscBool same;

    ierr = PetscStrcmp(name, pset->names[i].name, &same);CHKERRQ(ierr);
    if (same) {
      *polytope = pset->names[i].polytope;
      PetscFunctionReturn(0);
    }
  }
  *polytope = PETSCPOLYTOPE_NONE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetInsertName(PetscPolytopeSet pset, const char name[], PetscPolytope tope)
{
  PetscInt       index;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  index = pset->numNames++;
  if (index > pset->numNamesAlloc) {
    PetscPolytopeName *names;
    PetscInt newAlloc = PetscMax(8, pset->numNamesAlloc * 2);
    ierr = PetscCalloc1(newAlloc, &names);CHKERRQ(ierr);
    ierr = PetscArraycpy(names, pset->names, index);CHKERRQ(ierr);
    ierr = PetscFree(pset->names);CHKERRQ(ierr);
    pset->names = names;
  }
  pset->names[index].polytope = tope;
  ierr = PetscStrallocpy(name, &(pset->names[index].name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int RidgeSorterCompare(const void *a, const void *b)
{
  PetscInt i;
  const PetscInt *A = (const PetscInt *) a;
  const PetscInt *B = (const PetscInt *) b;

  /* compare ridge polytope */
  if (A[1] < B[1]) return -1;
  if (B[1] < A[1]) return 1;
  /* the same polytope (so same size): compare sorted vertices */
  for (i = 0; i < A[2]; i++) {
    if (A[3 + i] < B[3 + i]) return -1;
    if (B[3 + i] < A[3 + i]) return 1;
  }
  /* finally sort by facet */
  if (A[0] < B[0]) return -1;
  if (B[0] < A[0]) return 1;
  return 0;
}

static int SuppCompare(const void *a, const void *b)
{
  const PetscPolytopeSupp *A = (const PetscPolytopeSupp *) a;
  const PetscPolytopeSupp *B = (const PetscPolytopeSupp *) b;
  if (A->index < B->index) return -1;
  if (B->index < A->index) return 1;
  if (A->coneNumber < B->coneNumber) return -1;
  if (B->coneNumber < A->coneNumber) return 1;
  return 0;
}

/* facetsToRidges includes orientations */
/* ridgesToFacets includes cone numbers */
static PetscErrorCode PetscPolytopeSetComputeRidges(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt          numFacets, numVertices;
  const PetscPolytope *facets;
  const PetscInt *vertexOffsets;
  const PetscInt *facetsToVertices;
  PetscInt          i, r, maxRidgeSize, numFacetRidges, numRidges;
  PetscInt          sorterSize, count;
  PetscInt          *facetRidgeSorter;
  PetscInt          *ftro, *vwork, *rwork;
  PetscPolytopeCone *ftr;
  PetscPolytopeSupp *rtf;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  numFacets        = pData->numFacets;
  numVertices      = pData->numVertices;
  facets           = pData->facets;
  vertexOffsets    = pData->vertexOffsets;
  facetsToVertices = pData->facetsToVertices;
  ierr = PetscMalloc1(numFacets + 1,&ftro);CHKERRQ(ierr);
  pData->ridgeOffsets = ftro;
  ftro[0] = 0;
  for (i = 0, maxRidgeSize = 0, numFacetRidges = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    ftro[i+1] = ftro[i] + fData->numFacets;
    numFacetRidges += fData->numFacets;
    for (j = 0; j < fData->numFacets; j++) {
      PetscInt ridgeSize = fData->vertexOffsets[j+1] - fData->vertexOffsets[f];

      maxRidgeSize = PetscMax(maxRidgeSize,ridgeSize);
    }
  }
  if (numFacetRidges % 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No polytope has an odd number of facet ridges");
  pData->numRidges = numRidges = numFacetRidges / 2;
  ierr = PetscMalloc1(numFacetRidges, &ftr);CHKERRQ(ierr);
  pData->facetsToRidges = ftr;
  /* rtf is two (facets, cone number) pairs */
  ierr = PetscMalloc1(2 * numRidges, &rtf);CHKERRQ(ierr);
  pData->ridgesToFacets = rtf;
  sorterSize = 3 + maxRidgeSize; /* facetRidge, ridgePolytope, ridgeSize, ridgeVertices */
  ierr = PetscMalloc1(numFacetRidges * maxRidgeSize, &facetRidgeSorter);CHKERRQ(ierr);
  for (i = 0, r = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscInt          facetOffset = vertexOffsets[i];
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    for (j = 0; j < fData->numFacets; j++, r++) {
      PetscInt *ridge = &facetRidgeSorter[r*sorterSize];
      PetscInt *ridgeVertices = &ridge[3];
      PetscInt ridgeOffset = fData->vertexOffsets[j];
      PetscInt ridgeSize = fData->vertexOffsets[j+1] - ridgeOffset;
      PetscInt k;

      ridge[0] = r;
      ridge[1] = fData->facets[j];
      ridge[2] = ridgeSize;

      for (k = 0; k < ridgeSize; k++) {
        PetscInt v = fData->facetsToVertices[ridgeOffset + k];

        ridgeVertices[k] = facetsToVertices[facetOffset + v];
      }
      ierr = PetscSortInt(ridgeSize, ridgeVertices);CHKERRQ(ierr);
      for (k = ridgeSize; k < maxRidgeSize; k++) ridgeVertices[k] = -1;
    }
  }
  for (i = 0; i < numFacetRidges; i++) ftr[i].index = ftr[i].orientation = -1;
  for (i = 0; i < 2 * numRidges; i++) rtf[i].index = rtf[i].coneNumber = -1;
  /* all facetridges should occur as pairs: sort and detect them */
  qsort(facetRidgeSorter,(size_t) numFacetRidges, sorterSize * sizeof(PetscInt), RidgeSorterCompare);
  for (i = 1, count = 0; i < numFacetRidges; i++) {
    PetscInt  *ridgeA = &facetRidgeSorter[i*sorterSize];
    PetscInt  *ridgeB = &facetRidgeSorter[(i-1)*sorterSize];
    PetscBool same;

    ierr = PetscArraycmp(&ridgeA[1], &ridgeB[1], sorterSize-1, &same);CHKERRQ(ierr);
    if (!same) continue;
    if (ridgeA[0] == ridgeB[0]) break; /* cannot have one ridge appear twice in the cone of one facet */
    if (ftr[ridgeA[0]].index != -1 || ftr[ridgeB[0]].index != -1) break; /* cannot have more than two facets per ridge */
    ftr[ridgeA[0]].index = count;
    ftr[ridgeB[0]].index = count;
    rtf[2*count+0].index = ridgeA[0];
    rtf[2*count+1].index = ridgeB[0];
    count++;
  }
  ierr = PetscFree(facetRidgeSorter);CHKERRQ(ierr);
  if (i < numFacetRidges || count != numRidges) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope does not generate consistent ridges");
  /* convert facetRidge number to (facet, cone number) */
  for (i = 0, r = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    for (j = 0; j < fData->numFacets; j++, r++) {
      PetscInt ridge = ftr[r].index;
      PetscPolytopeSupp *ridgeData;

      if (ridge < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Polytope has unpaired ridge");
      ridgeData = &rtf[2 * ridge];
      if (ridgeData[0].index == r) {
        ridgeData[0].index = i;
        ridgeData[0].coneNumber = j;
      } else {
        if (ridgeData[1].index != r) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "ridgesToFacets / facetsToRidges mismatch");
        ridgeData[1].index = i;
        ridgeData[1].coneNumber = j;
      }
    }
  }
  /* re-sort rtf so that ridges appear in closure order */
  qsort(rtf,(size_t) numRidges, 2 * sizeof(PetscPolytopeSupp), SuppCompare);
  /* re-align ftr with re-sorted rtf */
  for (i = 0; i < numRidges; i++) {
    PetscPolytopeSupp *ridge = &rtf[2*i];
    PetscInt facetA = ridge[0].index;
    PetscInt coneA = ridge[0].coneNumber;
    PetscInt facetB = ridge[1].index;
    PetscInt coneB = ridge[1].coneNumber;

    ftr[ftro[facetA] + coneA].index = i;
    ftr[ftro[facetB] + coneB].index = i;
  }
  /* make sure that ridges line up in a way that makes sense:
   * the two orders of vertices from the opposing facets imply a
   * permutation of those vertices.  check that that symmetry is
   * valid for the polytope of the ridge */
  ierr = PetscMalloc1(numVertices, &vwork);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxRidgeSize, &rwork);CHKERRQ(ierr);
  for (i = 0, r = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscInt          facetOffset = vertexOffsets[i];
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    for (j = 0; j < fData->numFacets; j++, r++) {
      PetscInt          ridge = ftr[r].index;
      PetscPolytopeSupp *ridgeData;
      PetscInt          oppFacet, oppCone;
      PetscInt          k, orient;
      PetscPolytope     rtope = fData->facets[j];
      PetscPolytopeData rData = pset->polytopes[rtope];
      PetscPolytopeData oppData;
      PetscInt          ridgeOffset = fData->vertexOffsets[j];
      PetscInt          oppFacetOffset, oppRidgeOffset;
      PetscBool         isOrient;

      ridgeData = &rtf[2 * ridge];
      /* only compute symmetries from first side */
      if (ridgeData[0].index != r) continue;
      oppFacet = ridgeData[1].index;
      oppCone  = ridgeData[1].coneNumber;
      oppData  = pset->polytopes[oppFacet];
      oppFacetOffset = vertexOffsets[oppFacet];
      oppRidgeOffset = oppData->vertexOffsets[oppCone];
      ftr[r].orientation = 0; /* the orientation from the first facet is defined to be the identity */
      /* clear work */
      for (k = 0; k < numVertices; k++) vwork[k] = -1;
      for (k = 0; k < maxRidgeSize; k++) vwork[k] = -1;
      /* number vertices by the order they occur in the first facet numbering */
      for (k = 0; k < rData->numVertices; k++) {
        PetscInt rv = fData->facetsToVertices[ridgeOffset + k];
        PetscInt v = facetsToVertices[facetOffset + rv];

        vwork[v] = k;
      }
      /* gather that numbering from the perspective of the second facet */
      for (k = 0; k < rData->numVertices; k++) {
        PetscInt rv = oppData->facetsToVertices[oppRidgeOffset + k];
        PetscInt v = facetsToVertices[oppFacetOffset + rv];

        rwork[k] = vwork[v];
      }
      /* get the symmetry number */
      ierr = PetscPolytopeSetOrientationFromVertices(pset, rtope, rwork, &isOrient, &orient);CHKERRQ(ierr);
      if (!isOrient) break;
      ftr[ftro[oppFacet] + oppCone].orientation = orient;
    }
    if (j < fData->numFacets) break;
  }
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = PetscFree(vwork);CHKERRQ(ierr);
  if (i < numFacets) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Vertices of polytope ridge are permuted in a way that is not an orientation of the ridge");
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetComputeSigns(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt       numFacets, numRidges;
  const PetscPolytope *facets;
  const PetscInt *ridgeOffsets;
  const PetscPolytopeCone *facetsToRidges;
  const PetscPolytopeSupp *ridgesToFacets;
  PetscInt       i, rcount, fcount;
  PetscInt       *facetQueue, *ridgeQueue;
  PetscBool      *facetSeen, *ridgeSeen;
  PetscBool      *inward;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets        = pData->numFacets;
  if (!numFacets) PetscFunctionReturn(0);
  numRidges        = pData->numRidges;
  facets           = pData->facets;
  ridgeOffsets     = pData->ridgeOffsets;
  facetsToRidges   = pData->facetsToRidges;
  ridgesToFacets   = pData->ridgesToFacets;
  inward           = pData->facetsInward;
  ierr = PetscMalloc1(numRidges, &ridgeQueue);CHKERRQ(ierr);
  ierr = PetscCalloc1(numRidges, &ridgeSeen);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets, &facetQueue);CHKERRQ(ierr);
  ierr = PetscCalloc1(numFacets, &facetSeen);CHKERRQ(ierr);
  fcount = 1;
  facetQueue[0] = 0;
  facetSeen[0] = PETSC_TRUE;
  /* construct a breadth-first queue of ridges between facets,
   * so that we always test sign when at least one side
   * has its sign determined */
  for (i = 0, rcount = 0; i < numFacets; i++) {
    PetscInt f;
    PetscInt j;

    if (i == fcount) {
      ierr = PetscFree(facetSeen);CHKERRQ(ierr);
      ierr = PetscFree(facetQueue);CHKERRQ(ierr);
      ierr = PetscFree(ridgeSeen);CHKERRQ(ierr);
      ierr = PetscFree(ridgeQueue);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope is not strongly connected");
    }
    f = facetQueue[i];
    for (j = ridgeOffsets[f]; j < ridgeOffsets[f+1]; j++) {
      PetscInt r = facetsToRidges[j].index;
      PetscInt neigh;

      if (ridgeSeen[r]) continue;
      ridgeQueue[rcount++] = r;
      ridgeSeen[r] = PETSC_TRUE;
      if (ridgesToFacets[2*r].index == f) {
        neigh = ridgesToFacets[2*r+1].index;
      } else {
        neigh = ridgesToFacets[2*r].index;
      }
      if (!facetSeen[neigh]) {
        facetQueue[fcount++] = neigh;
        facetSeen[neigh] = PETSC_TRUE;
      }
    }
  }
  for (i = 0; i < numFacets; i++) facetSeen[i] = PETSC_FALSE;
  facetSeen[0] = PETSC_TRUE;
  for (i = 0; i < numRidges; i++) {
    PetscInt  r = ridgeQueue[i];
    PetscInt  f, g;
    PetscInt  fCone, gCone;
    PetscBool fSign, gSign;
    PetscInt  fOrient, gOrient;
    PetscInt  inwardg;
    PetscPolytopeData fData;
    PetscPolytopeData gData;

    f       = ridgesToFacets[2*r].index;
    fCone   = ridgesToFacets[2*r].coneNumber;
    fOrient = facetsToRidges[ridgeOffsets[f] + fCone].orientation;
    g       = ridgesToFacets[2*r+1].index;
    gCone   = ridgesToFacets[2*r+1].coneNumber;
    gOrient = facetsToRidges[ridgeOffsets[g] + gCone].orientation;
    if (!facetSeen[f]) {
      PetscInt swap;

      swap = f;
      f = g;
      g = swap;
      swap = fCone;
      fCone = gCone;
      gCone = swap;
      swap = fOrient;
      fOrient = gOrient;
      gOrient = swap;
    }
    if (!facetSeen[f]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Ridge between two unseen facets\n");
    fData = pset->polytopes[facets[f]];
    gData = pset->polytopes[facets[g]];
    ierr  = PetscPolytopeDataFacetSign(fData, fCone, &fSign);CHKERRQ(ierr);
    ierr  = PetscPolytopeDataFacetSign(gData, gCone, &gSign);CHKERRQ(ierr);
    fSign = fSign ^ (fOrient < 0);
    fSign = fSign ^ inward[f];
    gSign = gSign ^ (gOrient < 0);
    inwardg = gSign ^ fSign;
    if (!facetSeen[g]) {
      facetSeen[g] = PETSC_TRUE;
      inward[g] = inwardg;
    } else if (inward[g] != inwardg) {
      ierr = PetscFree(facetSeen);CHKERRQ(ierr);
      ierr = PetscFree(facetQueue);CHKERRQ(ierr);
      ierr = PetscFree(ridgeSeen);CHKERRQ(ierr);
      ierr = PetscFree(ridgeQueue);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope is not orientable\n");
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetAddSymmetry(PetscPolytopeSet pset, PetscPolytopeData pData, const PetscPolytopeCone *perm, const PetscPolytopeCone *permInv, PetscBT permOrient)
{
  PetscInt       numFacets, maxS;
  PetscInt       image0, orient0;
  PetscInt       invimage0, invorient0;
  PetscInt       q;
  PetscInt       foStart, foEnd, numFOs, id, invid, i, p;
  PetscInt       orientEndOrig;
  PetscPolytopeData fData;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets = pData->numFacets;
  maxS      = pData->maxFacetSymmetry;
  if (!numFacets) PetscFunctionReturn(0);
  fData   = pset->polytopes[pData->facets[0]];
  foStart = fData->orientStart;
  foEnd   = fData->orientEnd;
  numFOs  = foEnd - foStart;

  q = orientEndOrig = pData->orientEnd;

  image0  = perm[0].index;
  orient0 = perm[0].orientation;
  id = image0 * numFOs + (orient0 - foStart);
  if (PetscBTLookup(permOrient, id)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "A previously seen permutation slipped through the cracks");
  ierr = PetscBTSet(permOrient, id);CHKERRQ(ierr);
  for (i = 0; i < numFacets; i++) {
    PetscPolytopeData iData;
    if (perm[i].index < 0 || perm[i].index) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation has bad index");
    if (pData->facets[i] != pData->facets[perm[i].index]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation maps incompatible facets");
    iData = pset->polytopes[pData->facets[i]];
    if (perm[i].orientation < iData->orientStart || perm[i].orientation >= iData->orientEnd) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation has bad orientation");
    pData->orientsToFacetOrders[pData->orientEnd * numFacets + i] = perm[i];
  }
  pData->orientEnd++;

  invimage0  = permInv[0].index;
  invorient0 = perm[0].orientation;
  invid = invimage0 * numFOs + (invorient0 - foStart);
  if (invid != id) {
    if (PetscBTLookup(permOrient, invid)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "A previously seen inverse permutation slipped through the cracks");
    ierr = PetscBTSet(permOrient, invid);CHKERRQ(ierr);
    for (i = 0; i < numFacets; i++) {
      PetscPolytopeData iData;
      if (permInv[i].index < 0 || permInv[i].index) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation inverse has bad index");
      if (pData->facets[i] != pData->facets[permInv[i].index]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation maps incompatible facets");
      iData = pset->polytopes[pData->facets[i]];
      if (permInv[i].orientation < iData->orientStart || permInv[i].orientation >= iData->orientEnd) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation has bad orientation");
      pData->orientsToFacetOrders[pData->orientEnd * numFacets + i] = permInv[i];
    }
    pData->orientEnd++;
  }

  for (i = 0; i < numFacets; i++) {
    PetscPolytopeData iData;
    PetscInt o, oInvInv;
    if (perm[permInv[i].index].index != i) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "incorrect permutation inverse");
    iData = pset->polytopes[pData->facets[i]];
    oInvInv = iData->orientInverses[permInv[i].orientation];
    o = perm[permInv[i].index].orientation;
    if (o != oInvInv) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "incorrect permutation inverse orientations");
  }

  while (q < pData->orientEnd) {
    const PetscPolytopeCone *permNew = &pData->orientsToFacetOrders[q*numFacets];
    for (p = 0; p < q; p++) {
      const PetscPolytopeCone *permOld = &pData->orientsToFacetOrders[p*numFacets];
      PetscInt fComp = permOld[permNew[0].index].index;
      PetscInt oNew = permNew[0].orientation;
      PetscInt oOld = permOld[permNew[0].index].orientation;
      PetscInt oComp;
      PetscInt id;
      PetscInt prodIdx;
      PetscPolytopeCone *prod;

      if (permOld[0].index == 0 && permOld[0].orientation == 0-foStart) continue; /* permOld is the identity */
      ierr = PetscPolytopeSetOrientationCompose(pset, pData->facets[0], oNew, oOld, &oComp);CHKERRQ(ierr);
      prodIdx = pData->facetOrientations[maxS*fComp + oComp-foStart];
      if (prodIdx >= 0 && prodIdx < orientEndOrig) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation is new but product of existing permutations");
      if (fComp == 0 && oComp == 0-foStart) { /* permNew is the inverse of permOld */
        pData->orientInverses[p] = q;
        pData->orientInverses[q] = p;
      }
      if (prodIdx >= 0) continue; /* this product has been found before */
      id = fComp*numFOs + oComp;
      if (PetscBTLookup(permOrient, id)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "New product has been seen");
      ierr = PetscBTSet(permOrient, id);CHKERRQ(ierr);
      /* compute the full product */
      prod = &pData->orientsToFacetOrders[pData->orientEnd * numFacets];
      prod[0].index = fComp;
      prod[0].orientation = oComp;
      for (i = 1; i < numFacets; i++) {
        fComp = permOld[permNew[i].index].index;
        oNew  = permNew[i].orientation;
        oOld  = permOld[permNew[i].index].orientation;

        ierr = PetscPolytopeSetOrientationCompose(pset, pData->facets[i], oNew, oOld, &oComp);CHKERRQ(ierr);
        prod[i].index = fComp;
        prod[i].orientation = oComp;
      }
      /* update orbit projectors and facetOrientations */
      for (i = 0; i < numFacets; i++) {
        PetscInt representer;
        PetscInt f, o;
        PetscInt proj;

        f = prod[i].index;
        o = prod[i].orientation;
        proj = pData->orbitProjectors[f];
        representer = pData->orientsToFacetOrders[proj * numFacets + f].index;
        if (i <= representer) {
          PetscInt ioStart;
          PetscPolytopeData iData;

          iData = pset->polytopes[pData->facets[i]];
          ioStart = iData->orientStart;

          if (i < representer) { /* reset facetOrientations to get rid of data of previous representer */
            PetscInt j;
            pData->orbitProjectors[f] = pData->orientEnd;

            for (j = 0; j < maxS; j++) pData->facetOrientations[f*maxS + j] = PETSC_MIN_INT;
          }
          pData->facetOrientations[f*maxS + (o-ioStart)] = pData->orientEnd;
        }
      }
      pData->orientEnd++;
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetSignSymmetries(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt       o, f, numSyms, numFacets;
  PetscInt       maxS;
  PetscBool      *negative;
  PetscInt       *newIds, *newIdsInv;
  PetscInt       numNegative, count;
  PetscPolytopeCone *orientsToFacetOrders;
  PetscInt       *orientInverses;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets = pData->numFacets;
  if (!numFacets) PetscFunctionReturn(0);
  numSyms = pData->orientEnd;
  ierr = PetscCalloc1(numSyms, &negative);CHKERRQ(ierr);
  for (o = 0, numNegative = 0; o < pData->orientEnd; o++) {
    PetscInt f = pData->orientsToFacetOrders[o*numFacets].index;
    PetscInt fo = pData->orientsToFacetOrders[o*numFacets].orientation;

    negative[o] = (PetscBool) ((fo < 0) ^ (pData->facetsInward[0] == pData->facetsInward[f]));
    if (negative[o]) numNegative++;
  }
  if (!numNegative) {
    ierr = PetscFree(negative);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  pData->orientStart = -numNegative;
  pData->orientEnd = numSyms + pData->orientStart;
  ierr = PetscMalloc2(numSyms, &newIds, numSyms, &newIdsInv);CHKERRQ(ierr);
  for (o = 0, count = 0; o < numSyms; o++) {
    PetscInt newId = negative[o] ? (-numNegative + count++) : (o - count);

    newIds[o] = newId;
    newIdsInv[newId + numNegative] = o;
  }
  ierr = PetscMalloc1(numSyms * numFacets, &orientsToFacetOrders);CHKERRQ(ierr);
  for (o = 0; o < numSyms; o++) {
    for (f = 0; f < numFacets; f++) {
      orientsToFacetOrders[o * numFacets + f] = pData->orientsToFacetOrders[newIdsInv[o] * numFacets + f];
    }
  }
  ierr = PetscFree(pData->orientsToFacetOrders);CHKERRQ(ierr);
  pData->orientsToFacetOrders = orientsToFacetOrders;
  ierr = PetscMalloc1(numSyms, &orientInverses);CHKERRQ(ierr);
  for (o = 0; o < numSyms; o++) {
    orientInverses[o] = newIds[pData->orientInverses[newIdsInv[o]]];
  }
  ierr = PetscFree(pData->orientInverses);CHKERRQ(ierr);
  pData->orientInverses = orientInverses;
  maxS = pData->maxFacetSymmetry;
  for (f = 0; f < numFacets; f++) {
    pData->orbitProjectors[f] = newIds[pData->orbitProjectors[f]];
    for (o = 0; o < maxS; o++) {
      PetscInt ov = pData->facetOrientations[f*maxS+o];

      if (ov >= 0) {
        pData->facetOrientations[f*maxS+o] = newIds[ov];
      }
    }
  }
  ierr = PetscFree2(newIds, newIdsInv);CHKERRQ(ierr);
  ierr = PetscFree(negative);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetComputeSymmetries(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt       numFacets, numVertices;
  PetscInt       foStart, foEnd, numFOs, numFR, maxR, maxS;
  PetscPolytopeData fData;
  PetscInt       i, f, r, o;
  PetscBT        permOrient;
  PetscPolytopeCone *fCone, *perm, *permInv;
  PetscInt       *originQueue;
  PetscInt       fcount;
  PetscBool      *originSeen;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets = pData->numFacets;
  numVertices = pData->numVertices;
  /* every facet starts out with the identity */
  pData->orientStart = 0;
  pData->orientEnd   = 1;
  for (i = 0, maxS = 0; i < numFacets; i++) {
    fData = pset->polytopes[pData->facets[i]];
    maxS = PetscMax(maxS, fData->orientEnd - fData->orientEnd);
  }
  pData->maxFacetSymmetry = maxS;
  ierr = PetscMalloc1(numFacets * PetscMax(1, maxS), &(pData->orientsToFacetOrders));CHKERRQ(ierr);
  for (i = 0; i < numFacets; i++) {
    pData->orientsToFacetOrders[i].index       = i;
    pData->orientsToFacetOrders[i].orientation = 0;
  }
  ierr = PetscMalloc1(PetscMax(1, numFacets * maxS), &(pData->orientInverses));CHKERRQ(ierr);
  pData->orientInverses[0] = 0;
  ierr = PetscMalloc1(numFacets, &(pData->orbitProjectors));CHKERRQ(ierr);
  /* with just the identity, each facet is in its own orbit, so the identity is the projector
   * onto the orbit representer */
  for (i = 0; i < numFacets; i++) pData->orbitProjectors[i] = 0;
  ierr = PetscMalloc1(numFacets * maxS, &(pData->facetOrientations));CHKERRQ(ierr);
  /* with just the identity, a (facet, orientation) pair is valid only if the orientation is the
   * identity for that facets group, and it is always the identity */
  for (i = 0; i < numFacets; i++) {
    PetscPolytopeData iData = pset->polytopes[pData->facets[i]];
    PetscInt ioStart = iData->orientStart;
    PetscInt ioEnd = iData->orientEnd;
    PetscInt j;

    for (j = 0; j < ioEnd - ioStart; j++) {
      pData->facetOrientations[i * maxS + j] = (j == 0-ioStart) ? 0 : PETSC_MIN_INT;
    }
  }
  if (numFacets == 0 || numFacets == 1) {
    /* only identity */
    ierr = PetscMalloc1(numVertices, &(pData->orientsToVertexOrders));CHKERRQ(ierr);
    for (i = 0; i < numVertices; i++) pData->orientsToVertexOrders[i] = i;
    PetscFunctionReturn(0);
  }
  fData   = pset->polytopes[pData->facets[0]];
  foStart = fData->orientStart;
  foEnd   = fData->orientEnd;
  numFOs  = foEnd - foStart;
  ierr = PetscBTCreate(numFacets * numFOs, &permOrient);CHKERRQ(ierr);
  ierr = PetscBTMemzero(numFacets * numFOs, permOrient);CHKERRQ(ierr);
  /* we have already seen the identity */
  ierr = PetscBTSet(permOrient,0-foStart);CHKERRQ(ierr);
  for (f = 0, maxR = 0; f < numFacets; f++) maxR = PetscMax(maxR,pData->ridgeOffsets[f+1] - pData->ridgeOffsets[f]);
  ierr = PetscMalloc1(maxR, &fCone);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets, &originSeen);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets, &originQueue);CHKERRQ(ierr);
  ierr = PetscMalloc2(numFacets, &perm, numFacets, &permInv);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets, &pData->orbitProjectors);CHKERRQ(ierr);
  for (f = 0; f < maxS * numFacets; f++) pData->facetOrientations[f] = PETSC_MIN_INT;
  for (f = 0; f < numFacets; f++) pData->orbitProjectors[f] = PETSC_MIN_INT;
  for (f = 0; f < numFacets; f++) {
    if (pData->facets[f] != pData->facets[0]) continue; /* different facet polytopes, can't be a symmetry */
    for (o = foStart; o < foEnd; o++) {
      PetscInt id = f * numFOs + (o - foStart);
      PetscInt q, oInv;

      if (PetscBTLookup(permOrient, id)) continue; /* this orientation has been found */

      /* reset data */
      for (q = 0; q < numFacets; q++) {
        perm[q].index = -1;
        perm[q].orientation = -1;
        permInv[q].index = -1;
        permInv[q].orientation = -1;
        originSeen[q] = PETSC_FALSE;
      }

      ierr = PetscPolytopeDataOrientationInverse(fData, o, &oInv);CHKERRQ(ierr);
      perm[0].index = f;
      perm[0].orientation = o;
      permInv[f].index = 0;
      permInv[f].orientation = oInv;
      originSeen[0] = PETSC_TRUE;
      originQueue[0] = 0;
      fcount = 1;
      for (q = 0; q < numFacets; q++) {
        const PetscPolytopeCone *ftr0;
        const PetscPolytopeCone *ftrf;
        PetscPolytopeData oData;
        PetscInt origin;
        PetscInt target;
        PetscInt oo;
        if (q >= fcount) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Disconnected facet graph, should have been caught earlier");

        /* apply the orientation at the origin to the ridges, and use those to propagate to the facet neighbors */
        origin = originQueue[q];
        target = perm[origin].index;
        oo     = perm[origin].orientation;
        oData  = pset->polytopes[pData->facets[origin]];
        ierr   = PetscPolytopeDataFacetsFromOrientation(oData, oo, fCone);CHKERRQ(ierr);
        numFR  = oData->numFacets;
        ftr0   = &pData->facetsToRidges[pData->ridgeOffsets[origin]];
        ftrf   = &pData->facetsToRidges[pData->ridgeOffsets[target]];
        for (r = 0; r < numFR; r++) {
          PetscPolytope rtope;
          PetscPolytopeData rData;
          PetscPolytopeData nData;
          PetscInt rOrigin, rTarget;
          PetscInt n, nConeNum;
          PetscInt m, mConeNum;
          PetscInt oOrigin;
          PetscInt oOriginInv;
          PetscInt oTarget;
          PetscInt oComp;
          PetscInt oAction;
          PetscInt oN, oM, oMinv, oNM;
          PetscBool isOrient;

          rtope = oData->facets[r];
          rData = pset->polytopes[rtope];

          rOrigin = ftr0[r].index;
          oOrigin = ftr0[r].orientation;
          n = pData->ridgesToFacets[2*rOrigin].index;
          nConeNum = pData->ridgesToFacets[2*rOrigin].coneNumber;
          if (n == origin) {
            n = pData->ridgesToFacets[2*rOrigin+1].index;
            nConeNum = pData->ridgesToFacets[2*rOrigin+1].coneNumber;
          }

          rTarget = ftrf[fCone[r].index].index;
          oTarget = ftrf[fCone[r].index].orientation;
          m = pData->ridgesToFacets[2*rTarget].index;
          mConeNum = pData->ridgesToFacets[2*rTarget].coneNumber;
          if (m == target) {
            m = pData->ridgesToFacets[2*rTarget+1].index;
            mConeNum = pData->ridgesToFacets[2*rTarget+1].coneNumber;
          }
          if (pData->facets[m] != pData->facets[n]) break; /* this is not a compatible orientation: it maps different types of facets onto each other */
          nData = pset->polytopes[pData->facets[n]];

          oN = pData->facetsToRidges[pData->ridgeOffsets[n] + nConeNum].orientation;
          oM = pData->facetsToRidges[pData->ridgeOffsets[m] + mConeNum].orientation;

          oAction = fCone[r].orientation;
          ierr = PetscPolytopeDataOrientationInverse(rData, oOrigin, &oOriginInv);CHKERRQ(ierr);
          ierr = PetscPolytopeDataOrientationInverse(rData, oM, &oMinv);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oOriginInv, oN, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oAction, oComp, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oTarget, oComp, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oMinv, oComp, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationFromFacet(pset, n, nConeNum, mConeNum, oComp, &isOrient, &oNM);CHKERRQ(ierr);
          if (!isOrient) break; /* TODO: can this happen ? */
          if (!originSeen[n]) {
            PetscInt oNMinv;
            originSeen[n] = PETSC_TRUE;

            perm[n].index = m;
            perm[n].orientation = oNM;
            ierr = PetscPolytopeDataOrientationInverse(nData, oNM, &oNMinv);CHKERRQ(ierr);
            permInv[m].index = n;
            permInv[m].orientation = n;
            fcount++;
          } else {
            if (perm[n].index != m || perm[n].orientation != oNM) break; /* different dictates from different ridges, orientation impossible */
          }
        }
        if (r < numFR) break;
      }
      if (q < numFacets) continue;
      ierr = PetscPolytopeSetAddSymmetry(pset, pData, perm, permInv, permOrient);CHKERRQ(ierr);
    }
  }
  ierr = PetscPolytopeSetSignSymmetries(pset, pData);CHKERRQ(ierr);
  ierr = PetscFree2(perm,permInv);CHKERRQ(ierr);
  ierr = PetscFree(originQueue);CHKERRQ(ierr);
  ierr = PetscFree(originSeen);CHKERRQ(ierr);
  ierr = PetscFree(fCone);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&permOrient);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define CHKPOLYTOPEERRQ(pData,ierr) if (ierr) {PetscErrorCode _ierr = PetscPolytopeDataDestroy(&(pData));CHKERRQ(_ierr);CHKERRQ(ierr);}

static PetscErrorCode PetscPolytopeSetInsert(PetscPolytopeSet pset, const char name[], PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscPolytope *polytope)
{
  PetscInt          i, dim;
  PetscPolytope     existing, id;
  PetscPolytopeData pData;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (numFacets < 0 || numVertices < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Attempt to create polytope with negative sizes (%D facets, %D vertices)\n", numFacets, numVertices);
  ierr = PetscPolytopeSetGetPolytope(pset, name, &existing);CHKERRQ(ierr);
  if (existing != PETSCPOLYTOPE_NONE) {
    PetscBool same;

    ierr = PetscPolytopeDataCompare(pset->polytopes[existing], numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &same);CHKERRQ(ierr);
    if (same) {
      *polytope = existing;
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to insert polytope %s with different data than existing polytope with that name\n", name);
  }
  for (id = 0; id < pset->numPolytopes; id++) {
    PetscBool same;

    ierr = PetscPolytopeDataCompare(pset->polytopes[id], numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &same);CHKERRQ(ierr);
    if (same) break;
  }
  if (id < pset->numPolytopes) {
    ierr = PetscPolytopeSetInsertName(pset, name, id);CHKERRQ(ierr);
    *polytope = id;
    PetscFunctionReturn(0);
  }
  { /* make sure this polytope has a consistent dimension (recursively) */
    PetscInt minDim = PETSC_MAX_INT;
    PetscInt maxDim = PETSC_MIN_INT;

    for (i = 0; i < numFacets; i++) {
      PetscPolytopeData fdata;
      PetscInt          numVertices;
      PetscPolytope     f = facets[i];

      if (f < 0 || f > pset->numPolytopes) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Polytope facet %D has id %D which is unknown to the polytope set\n", i, f);
      fdata = pset->polytopes[f];
      numVertices = vertexOffsets[i + 1] - vertexOffsets[i];
      if (numVertices != fdata->numVertices) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Polytope facet %D has id %D but %D vertices != %D\n", i, f, numVertices, fdata->numVertices);
      minDim = PetscMin(minDim, fdata->dim);
      maxDim = PetscMin(maxDim, fdata->dim);
    }
    if (!numFacets) minDim = maxDim = -2;
    if (minDim != maxDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope facets have inconsistent dimensions in [%D,%D]", minDim, maxDim);
    dim = minDim + 1;
  }
  { /* make sure the vertex order is consistent with the closure order */
    PetscInt   *closureVertices;
    PetscBool  *seen;
    PetscInt   count;
    PetscBool  inOrder;
    char       suggestion[256] = {0};

    ierr = PetscMalloc1(numVertices, &closureVertices);CHKERRQ(ierr);
    ierr = PetscCalloc1(numVertices, &seen);CHKERRQ(ierr);
    for (i = 0, count = 0; i < numFacets; i++) {
      PetscInt j;

      for (j = vertexOffsets[i]; j < vertexOffsets[i + 1]; j++) {
        PetscInt v = facetsToVertices[j];

        if (!seen[v]) {
          seen[v] = PETSC_TRUE;
          closureVertices[count++] = v;
        }
      }
    }
    if (count != numVertices) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope does not touch all vertices");
    for (i = 0; i < numFacets; i++) if (closureVertices[i] != i) break;
    inOrder = (PetscBool) (i == numFacets);
    if (!inOrder) {
      PetscInt origLength, length;
      char *top;

      length = origLength = sizeof(suggestion);
      top = suggestion;
      for (i = 0; i < numVertices; i++) {
        if (i < numVertices-1) {
          ierr = PetscSNPrintf(top,length,"%D,",closureVertices[i]);CHKERRQ(ierr);
        } else {
          ierr = PetscSNPrintf(top,length,"%D",closureVertices[i]);CHKERRQ(ierr);
        }
        while ((top - suggestion) < (origLength - 1) && *top != '\0') {
          top++;
          length--;
        }
      }
    }
    ierr = PetscFree(seen);CHKERRQ(ierr);
    ierr = PetscFree(closureVertices);CHKERRQ(ierr);
    if (!inOrder) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Polytope vertices not in closure order: suggested reordering [%s]\n", suggestion);
  }
  ierr = PetscPolytopeDataCreate(dim, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &pData);CHKERRQ(ierr);
  /* I want an incorrectly specified polytope to be recoverable, so I try to tear down the partially created polytope
   * if these two routines fail */
  ierr = PetscPolytopeSetComputeRidges(pset, pData);CHKPOLYTOPEERRQ(pData,ierr);
  ierr = PetscPolytopeSetComputeSigns(pset, pData);CHKPOLYTOPEERRQ(pData,ierr);
  /* Any failure in compute symmetries should be a PLIB failure and not recoverable */
  ierr = PetscPolytopeSetComputeSymmetries(pset, pData);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopesDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopeSetDestroy(&PetscPolytopes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeInsert(const char name[], PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscPolytope *polytope)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PetscPolytopes) {
    ierr = PetscPolytopeSetCreate(&PetscPolytopes);CHKERRQ(ierr);
    ierr = PetscRegisterFinalize(PetscPolytopesDestroy);CHKERRQ(ierr);
  }
  ierr = PetscPolytopeSetInsert(PetscPolytopes, name, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, polytope);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
