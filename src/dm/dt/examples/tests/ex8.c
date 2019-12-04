static char help[] = "Tests PetscPolytope.\n\n";

#include <petscdt.h>

static PetscErrorCode PetscPolytopeInsertCheck(const char name[], PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscBool isRefiner, PetscPolytope *polytope)
{
  PetscPolytope  p, pagain, pbyname;
  char           namedup[256];
  PetscInt       numF, numV;
  const PetscPolytope *f;
  const PetscInt *v;
  const PetscInt *ftv;
  const PetscBool *in;
  PetscBool      same;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!isRefiner) {ierr = PetscPolytopeInsert(name, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &p);CHKERRQ(ierr);}
  else            {ierr = PetscPolytopeInsertRefinement(name, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, &p);CHKERRQ(ierr);}
  ierr = PetscSNPrintf(namedup, 256, "%s-dup", name);CHKERRQ(ierr);
  if (!isRefiner) {ierr = PetscPolytopeInsert(namedup, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &pagain);CHKERRQ(ierr);}
  else            {ierr = PetscPolytopeInsertRefinement(namedup, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, &pagain);CHKERRQ(ierr);}
  if (pagain != p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Identical polytopes not identified\n");
  ierr = PetscPolytopeGetByName(name, &pbyname);CHKERRQ(ierr);
  if (pbyname != p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Name recall error\n");
  pbyname = PETSCPOLYTOPE_NONE;
  ierr = PetscPolytopeGetByName(namedup, &pbyname);CHKERRQ(ierr);
  if (pbyname != p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Name recall error\n");
  ierr = PetscPolytopeGetData(p, &numF, &numV, &f, &v, &ftv, &in);CHKERRQ(ierr);
  if (numF != numFacets || numV != numVertices) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect counts returned for polytope\n");
  ierr = PetscArraycmp(facets, f, numFacets, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Non-matching returned facets\n");
  if (numFacets) {
    ierr = PetscArraycmp(vertexOffsets, v, numFacets+1, &same);CHKERRQ(ierr);
    if (!same) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Non-matching returned vertexOffsets\n");
    if (in[0] != firstFacetInward) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "first facet sign changed\n");
  } else if (v[0] != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Non-sensical vertexOffsets returned\n");
  ierr = PetscArraycmp(facetsToVertices, ftv, v[numFacets], &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Non-matching returned facetsToVertices\n");
  *polytope = p;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeInsertCheckSignsSymmetry(const char name[], PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[],
                                                            const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward,
                                                            PetscBool isRefiner, const PetscBool signs[], PetscInt oStart, PetscInt oEnd, PetscPolytope *polytope)
{
  PetscInt f, coStart, coEnd, o, w, wo, v;
  const PetscBool *facetsInward;
  PetscInt       *ov, *wv, *wov;
  PetscInt       *of, *wf, *wof;
  PetscInt       *ofo, *wfo, *wofo;
  PetscPolytope  p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopeInsertCheck(name, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, isRefiner, &p);CHKERRQ(ierr);
  ierr = PetscPolytopeGetData(p, NULL, NULL, NULL, NULL, NULL, &facetsInward);CHKERRQ(ierr);
  for (f = 0; f < numFacets; f++) if (facetsInward[f] != (signs ? signs[f] : PETSC_FALSE)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s does not have correct facet signs\n", name);
  ierr = PetscPolytopeGetOrientationRange(p, &coStart, &coEnd);CHKERRQ(ierr);
  if (coStart != oStart || coEnd != oEnd) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of %s\n", name);
  ierr = PetscMalloc3(numVertices, &ov, numVertices, &wv, numVertices, &wov);CHKERRQ(ierr);
  ierr = PetscMalloc3(numFacets, &of, numFacets, &wf, numFacets, &wof);CHKERRQ(ierr);
  ierr = PetscMalloc3(numFacets, &ofo, numFacets, &wfo, numFacets, &wofo);CHKERRQ(ierr);
  for (o = oStart; o < oEnd; o++) {
    PetscBool isOrient;
    PetscInt  ocheck, oinv;

    ierr = PetscPolytopeOrientationInverse(p, o, &oinv);CHKERRQ(ierr);
    ierr = PetscPolytopeOrientationInverse(p, oinv, &ocheck);CHKERRQ(ierr);
    if (ocheck != o) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s orientation inverse failure", name);

    ierr = PetscPolytopeOrientVertices(p, o, ov);CHKERRQ(ierr);
    ierr = PetscPolytopeOrientationFromVertices(p, ov, &isOrient, &ocheck);CHKERRQ(ierr);
    if (!isOrient || ocheck != o) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s orient -> vertex -> orient failure", name);

    ierr = PetscPolytopeOrientFacets(p, o, of, ofo);CHKERRQ(ierr);
    for (f = 0; f < numFacets; f++) {
      ierr = PetscPolytopeOrientationFromFacet(p, f, of[f], ofo[f], &isOrient, &ocheck);CHKERRQ(ierr);
      if (!isOrient || ocheck != o) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s orient -> (facet, o) -> orient failure", name);
    }
    for (w = oStart; w < oEnd; w++) {
      ierr = PetscPolytopeOrientationCompose(p, w, o, &wo);CHKERRQ(ierr);

      ierr = PetscPolytopeOrientVertices(p, w, wv);CHKERRQ(ierr);
      ierr = PetscPolytopeOrientVertices(p, wo, wov);CHKERRQ(ierr);
      for (v = 0; v < numVertices; v++) {
        if (ov[wv[v]] != wov[v]) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s vertex permutation composition incompatibility", name);
      }

      ierr = PetscPolytopeOrientFacets(p, w, wf, wfo);CHKERRQ(ierr);
      ierr = PetscPolytopeOrientFacets(p, wo, wof, wofo);CHKERRQ(ierr);

      for (f = 0; f < numFacets; f++) {
        PetscInt wofocheck;

        if (of[wf[f]] != wof[f]) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s facet permutation composition incompatibility", name);
        ierr = PetscPolytopeOrientationCompose(facets[f], wfo[f], ofo[wf[f]], &wofocheck);CHKERRQ(ierr);
        if (wofocheck != wofo[f]) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s facet orientation composition incompatibility", name);
      }
    }
  }
  ierr = PetscFree3(ofo, wfo, wofo);CHKERRQ(ierr);
  ierr = PetscFree3(of, wf, wof);CHKERRQ(ierr);
  ierr = PetscFree3(ov, wv, wov);CHKERRQ(ierr);
  *polytope = p;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscPolytope  null, no_point, vertex, edge, tri, noncyctri, quad, noncycquad,
                 tet, noncyctet, hex, pent, wedge, pyr, dodec, oct, rhomb, icos,
                 pchor, tess, ortho, oplex;
  PetscPolytope  edge_bisect, edge_trisect, tri_refine_regular, tri_refine_sierpinski, tri_refine_quads,
                 quad_refine_aniso, quad_refine_morton, quad_refine_hilbert, quad_refine_peano,
                 tet_refine_regular, hex_refine_morton, hex_refine_hilbert;
  PetscInt       oStart, oEnd;
  PetscErrorCode ierr, testerr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscPolytopeInsertCheck("null", 0, 0, NULL, NULL, NULL, PETSC_FALSE, PETSC_FALSE, &null);CHKERRQ(ierr);
  ierr = PetscPolytopeGetByName("not-a-point", &no_point);CHKERRQ(ierr);
  if (no_point != PETSCPOLYTOPE_NONE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unrecognized name did not return PETSCPOLYTOPE_NONE\n");
  ierr = PetscPolytopeGetOrientationRange(null, &oStart, &oEnd);CHKERRQ(ierr);
  if (oStart != 0 || oEnd != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of null point");

  {
    PetscInt vertexOffsets[2] = {0,0};
    PetscPolytope inv_vertex;
    PetscBool signs[1] = {PETSC_FALSE};
    PetscBool inv_signs[1] = {PETSC_TRUE};

    ierr = PetscPolytopeInsertCheckSignsSymmetry("vertex", 1, 0, &null, vertexOffsets, NULL, PETSC_FALSE, PETSC_FALSE, signs, 0, 1, &vertex);CHKERRQ(ierr);
    ierr = PetscPolytopeInsertCheckSignsSymmetry("vertex-inverted", 1, 0, &null, vertexOffsets, NULL, PETSC_TRUE, PETSC_FALSE, inv_signs, 0, 1, &inv_vertex);CHKERRQ(ierr);
    if (inv_vertex == vertex) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverted vertex the same as vertex");CHKERRQ(ierr);
  }

  {
    PetscPolytope facets[2];
    PetscInt      vertexOffsets[3] = {0,1,2};
    PetscInt      facetsToVertices[2] = {0, 1};
    PetscBool     signs[2] = {PETSC_TRUE, PETSC_FALSE};
    PetscBool     inv_signs[2] = {PETSC_FALSE, PETSC_TRUE};
    PetscPolytope inv_edge;

    facets[0] = facets[1] = vertex;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("edge", 2, 2, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_FALSE, signs, -1, 1, &edge);CHKERRQ(ierr);
    ierr = PetscPolytopeInsertCheckSignsSymmetry("edge-inverted", 2, 2, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, inv_signs, -1, 1, &inv_edge);CHKERRQ(ierr);
    if (inv_edge == edge) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverted edge the same as edge");
  }

  { /* bisected edge */
    PetscPolytope facets[3];
    PetscInt      vertexOffsets[4] = {0,2,4,6};
    PetscInt      facetsToVertices[6] = {0,1, 0,2, 2,1};
    PetscBool     signs[3] = {PETSC_TRUE, PETSC_FALSE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = edge;
    ierr = PetscPolytopeInsertCheckSignsSymmetry("edge-bisected", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -1, 1, &edge_bisect);CHKERRQ(ierr);
  }

  { /* trisected edge */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,2,4,6,8};
    PetscInt      facetsToVertices[8] = {0,1, 0,2, 2,3, 3,1};
    PetscBool     signs[4] = {PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = facets[3] = edge;
    ierr = PetscPolytopeInsertCheckSignsSymmetry("edge-trisected", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -1, 1, &edge_trisect);CHKERRQ(ierr);
  }

  { /* try and fail to create a mixed dimension polytope */
    PetscPolytope facets[2];
    PetscInt      vertexOffsets[3] = {0,1,3};
    PetscInt      facetsToVertices[3] = {0,1,2};
    PetscLogDouble mempre, mempost;

    facets[0] = vertex;
    facets[1] = edge;

    ierr = PetscMemoryGetCurrentUsage(&mempre);CHKERRQ(ierr);
    ierr = PetscPushErrorHandler(PetscReturnErrorHandler,NULL);CHKERRQ(ierr);
    testerr = PetscPolytopeInsert("mixed-triangle", 2, 3, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, &edge);
    ierr = PetscPopErrorHandler();
    if (testerr != PETSC_ERR_ARG_WRONG) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not catch mixed dimension polytope");
    ierr = PetscMemoryGetCurrentUsage(&mempost);CHKERRQ(ierr);
    if (mempre != mempost) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Recoverable error not memory neutral");
  }

  { /* cyclic triangle */
    PetscPolytope facets[3];
    PetscInt      vertexOffsets[4] = {0,2,4,6};
    PetscInt      facetsToVertices[6] = {0,1, 1,2, 2,0};

    facets[0] = facets[1] = facets[2] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("triangle", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -3, 3, &tri);CHKERRQ(ierr);
  }

  { /* non-cyclic triangle */
    PetscPolytope facets[3];
    PetscInt      vertexOffsets[4] = {0,2,4,6};
    PetscInt      facetsToVertices[6] = {0,1, 0,2, 1,2};
    PetscBool     signs[3] = {PETSC_FALSE, PETSC_TRUE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("non-cyclic-triangle", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, signs, -3, 3, &noncyctri);CHKERRQ(ierr);
  }

  { /* regularly refined triangle */
    PetscPolytope facets[8];
    PetscInt      vertexOffsets[9] = {0,3,6,9,12,15,18,21,24};
    PetscInt      facetsToVertices[24] = {
                                         0,1,2, /* parent */
                                         0,1,3, /* refined edge */
                                         1,2,4, /* refined edge */
                                         2,0,5, /* refined edge */
                                         0,3,5, /* child */
                                         3,1,4, /* child */
                                         5,4,2, /* child */
                                         4,5,3, /* child */
                                         };
    PetscInt      f;
    PetscBool     signs[8];

    for (f = 0; f < 4; f++) signs[f] = PETSC_TRUE;
    for (f = 4; f < 8; f++) signs[f] = PETSC_FALSE;
    facets[0] = tri; /* parent */
    facets[1] = facets[2] = facets[3] = edge_bisect; /* refined edges */
    facets[4] = facets[5] = facets[6] = facets[7] = tri; /* children */
    ierr = PetscPolytopeInsertCheckSignsSymmetry("triangle-refined-regular", 8, 6, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -3, 3, &tri_refine_regular);CHKERRQ(ierr);
  }

  { /* Sierpinski bisected triangle */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,3,6,9,12};
    PetscInt      facetsToVertices[12] = {
                                         0,1,2, /* parent */
                                         1,2,3, /* refined edge */
                                         1,3,0, /* child */
                                         0,3,2, /* child */
                                         };
    PetscBool     signs[4] = {PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE};

    facets[0] = tri; /* parent */
    facets[1] = edge_bisect; /* refined edges */
    facets[2] = facets[3] = tri; /* children */
    ierr = PetscPolytopeInsertCheckSignsSymmetry("triangle-refined-sierpinski", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -1, 1, &tri_refine_sierpinski);CHKERRQ(ierr);
  }

  { /* cyclic quadrilateral */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,2,4,6,8};
    PetscInt      facetsToVertices[8] = {0,1, 1,2, 2,3, 3,0};

    facets[0] = edge;
    facets[1] = edge;
    facets[2] = edge;
    facets[3] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -4, 4, &quad);CHKERRQ(ierr);
  }

  { /* non-cyclic quadrilateral */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,2,4,6,8};
    PetscInt      facetsToVertices[8] = {0,1, 2,3, 0,2, 1,3};
    PetscBool     signs[4] = {PETSC_FALSE, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = facets[3] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("non-cyclic-quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, signs, -4, 4, &noncycquad);CHKERRQ(ierr);
  }

  { /* anisotropically refined quadrilateral */
    PetscPolytope facets[5];
    PetscInt      vertexOffsets[6] = {0,4,7,10,14,18};
    PetscInt      facetsToVertices[18] = {
                                         0,1,2,3, /* parent */
                                         1,2,4,   /* refined edge */
                                         3,0,5,   /* refined edge */
                                         0,1,4,5, /* child */
                                         5,4,2,3, /* child */
                                         };
    PetscInt      f;
    PetscBool     signs[5];

    for (f = 0; f < 3; f++) signs[f] = PETSC_TRUE;
    for (f = 3; f < 5; f++) signs[f] = PETSC_FALSE;
    facets[0] = quad;                    /* parent */
    facets[1] = facets[2] = edge_bisect; /* refined edges */
    facets[3] = facets[4] = quad;        /* children */
    ierr = PetscPolytopeInsertCheckSignsSymmetry("quadrilateral-refined-anisotropically", 5, 6, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -2, 2, &quad_refine_aniso);CHKERRQ(ierr);
  }

  { /* morton */
    PetscPolytope facets[9];
    PetscInt      vertexOffsets[10] = {0,4,7,10,13,16,20,24,28,32};
    PetscInt      facetsToVertices[32] = {
                                         0,1,2,3, /* parent */
                                         0,1,4,   /* refined edge */
                                         1,2,5,   /* refined edge */
                                         2,3,6,   /* refined edge */
                                         3,0,7,   /* refined edge */
                                         0,4,8,7, /* child */
                                         4,1,5,8, /* child */
                                         7,8,6,3, /* child */
                                         8,5,2,6, /* child */
                                         };
    PetscInt      f;
    PetscBool     signs[9];

    for (f = 0; f < 5; f++) signs[f] = PETSC_TRUE;
    for (f = 5; f < 9; f++) signs[f] = PETSC_FALSE;
    facets[0] = quad;                    /* parent */
    facets[1] = facets[2] = facets[3] = facets[4] = edge_bisect; /* refined edges */
    facets[5] = facets[6] = facets[7] = facets[8] = quad;        /* children */
    ierr = PetscPolytopeInsertCheckSignsSymmetry("quadrilateral-refined-morton", 9, 9, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -4, 4, &quad_refine_morton);CHKERRQ(ierr);
  }

  { /* hilbert */
    PetscPolytope facets[9];
    PetscInt      vertexOffsets[10] = {0,4,7,10,13,16,20,24,28,32};
    PetscInt      facetsToVertices[32] = {
                                         0,1,2,3, /* parent */
                                         0,1,4,   /* refined edge */
                                         1,2,5,   /* refined edge */
                                         2,3,6,   /* refined edge */
                                         3,0,7,   /* refined edge */
                                         0,7,8,4, /* child */
                                         4,1,5,8, /* child */
                                         8,5,2,6, /* child */
                                         6,8,7,3, /* child */
                                         };
    PetscInt      f;
    PetscBool     signs[9];

    for (f = 0; f < 5; f++) signs[f] = PETSC_TRUE;
    signs[5] = signs[8] = PETSC_TRUE; /* first and last quads are flipped */
    signs[6] = signs[7] = PETSC_FALSE;
    facets[0] = quad;                    /* parent */
    facets[1] = facets[2] = facets[3] = facets[4] = edge_bisect; /* refined edges */
    facets[5] = facets[6] = facets[7] = facets[8] = quad;        /* children */
    ierr = PetscPolytopeInsertCheckSignsSymmetry("quadrilateral-refined-hilbert", 9, 9, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -4, 4, &quad_refine_hilbert);CHKERRQ(ierr);
  }

  { /* hilbert */
    PetscPolytope facets[14];
    PetscInt      vertexOffsets[15] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56};
    PetscInt      facetsToVertices[56] = {
                                          0, 1, 2, 3, /* parent */
                                          0, 1, 4, 5, /* refined edge */
                                          1, 2, 6, 7, /* refined edge */
                                          2, 3, 8, 9, /* refined edge */
                                          3, 0,10,11, /* refined edge */
                                          0, 4,12,11, /* child */
                                         12, 4, 5,13, /* child */
                                          5, 1, 6,13, /* child */
                                          6, 7,14,13, /* child */
                                         14,15,12,13, /* child */
                                         12,15,10,11, /* child */
                                         10,15, 9, 3, /* child */
                                          9,15,14, 8, /* child */
                                         14, 7, 2, 8, /* child */
                                         };
    PetscInt      f;
    PetscBool     signs[14];

    for (f = 0; f < 5; f++) signs[f] = PETSC_TRUE;
    for (f = 5; f < 14; f++) signs[f] = PETSC_FALSE;
    facets[0] = quad;                    /* parent */
    facets[1] = facets[2] = facets[3] = facets[4] = edge_trisect; /* refined edges */
    for (f = 5; f < 14; f++) facets[f] = quad; /* children */
    ierr = PetscPolytopeInsertCheckSignsSymmetry("quadrilateral-refined-peano", 14, 16, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -4, 4, &quad_refine_peano);CHKERRQ(ierr);
  }

  { /* triangle into quads */
    PetscPolytope facets[7];
    PetscInt      vertexOffsets[8] = {0,3,6,9,12,16,20,24};
    PetscInt      facetsToVertices[24] = {
                                         0,1,2,   /* parent */
                                         0,1,3,   /* refined edge */
                                         1,2,4,   /* refined edge */
                                         2,0,5,   /* refined edge */
                                         0,3,6,5, /* child */
                                         1,4,6,3, /* child */
                                         2,5,6,4, /* child */
                                         };
    PetscInt      f;
    PetscBool     signs[5];

    for (f = 0; f < 4; f++) signs[f] = PETSC_TRUE;
    for (f = 4; f < 7; f++) signs[f] = PETSC_FALSE;
    facets[0] = tri;                    /* parent */
    facets[1] = facets[2] = facets[3] = edge_bisect; /* refined edges */
    facets[4] = facets[5] = facets[6] = quad;        /* children */
    ierr = PetscPolytopeInsertCheckSignsSymmetry("triangle-refined-quadrilaterals", 7, 7, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -3, 3, &tri_refine_quads);CHKERRQ(ierr);
  }

  { /* cyclic pentagon */
    PetscPolytope facets[5];
    PetscInt      vertexOffsets[6] = {0,2,4,6,8,10};
    PetscInt      facetsToVertices[10] = {0,1, 1,2, 2,3, 3,4, 4,0};

    facets[0] = facets[1] = facets[2] = facets[3] = facets[4] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("pentagon", 5, 5, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -5, 5, &pent);CHKERRQ(ierr);
  }

  { /* tetrahedron */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,3,6,9,12};
    PetscInt      facetsToVertices[12] = {0,1,2, 0,3,1, 0,2,3, 2,1,3};

    facets[0] = facets[1] = facets[2] = facets[3] = tri;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("tetrahedron", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -12, 12, &tet);CHKERRQ(ierr);
  }

  { /* non-cyclic tetrahedron */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,3,6,9,12};
    PetscInt      facetsToVertices[12] = {0,1,2, 0,1,3, 0,2,3, 1,2,3};
    PetscBool     signs[4] = {PETSC_TRUE, PETSC_FALSE, PETSC_TRUE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = facets[3] = noncyctri;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("non-cyclic tetrahedron", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_FALSE, signs, -12, 12, &noncyctet);CHKERRQ(ierr);
  }

  { /* regularly refined tetrahedron */
    PetscPolytope facets[13];
    PetscInt      vertexOffsets[14] = {0,4,10,16,22,28,32,36,40,44,48,52,56,60};
    PetscInt      facetsToVertices[60] = {
                                         0,1,2,3, /* parent */
                                         0,1,2,4,5,6, /* refined face */
                                         0,3,1,7,8,4, /* refined face */
                                         0,2,3,6,9,7, /* refined face */
                                         2,1,3,5,8,9, /* refined face */
                                         0,4,6,7, /* child */
                                         4,1,5,8, /* child */
                                         6,5,2,9, /* child */
                                         7,8,9,3, /* child */
                                         6,7,4,9, /* child */
                                         8,5,4,9, /* child */
                                         9,4,6,5, /* child */
                                         9,4,8,7, /* child */
                                         };
    PetscInt      f;
    PetscBool     signs[13];

    for (f = 0; f < 5; f++)  signs[f] = PETSC_TRUE;
    for (f = 5; f < 13; f++) signs[f] = PETSC_FALSE;

    facets[0] = tet;
    for (f = 1; f < 5; f++)  facets[f] = tri_refine_regular;
    for (f = 5; f < 13; f++) facets[f] = tet;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("tetrahedron-refined-regular", 13, 10, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -4, 4, &tet_refine_regular);CHKERRQ(ierr);
  }

  { /* hexahedron */
    PetscPolytope facets[6];
    PetscInt      vertexOffsets[7] = {0,4,8,12,16,20,24};
    PetscInt      facetsToVertices[24] = {0,1,2,3, 4,5,6,7, 0,3,5,4, 2,1,7,6, 3,2,6,5, 0,4,7,1};

    facets[0] = facets[1] = facets[2] = facets[3] = facets[4] = facets[5] = quad;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("hexahedron", 6, 8, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -24, 24, &hex);CHKERRQ(ierr);
  }

  { /* morton refined hexahedron */
    PetscPolytope facets[15];
    PetscInt      vertexOffsets[16] = {0,8, 17,26,35,44,53,62, 70,78,86,94,102,110,118,126};
    PetscInt      facetsToVertices[126] = {
                                           0, 1, 2, 3, 4, 5, 6, 7,    /* parent */
                                           0, 1, 2, 3, 8, 9,10,11,12, /* refined face */
                                           4, 5, 6, 7,13,14,15,16,17, /* refined face */
                                           0, 3, 5, 4,11,18,13,19,20, /* refined face */
                                           2, 1, 7, 6, 9,21,15,22,23, /* refined face */
                                           3, 2, 6, 5,10,22,14,18,24, /* refined face */
                                           0, 4, 7, 1,19,16,21, 8,25, /* refined face */
                                           0, 8,12,11,19,20,26,25,    /* child */
                                          11,12,10, 3,20,18,24,26,    /* child */
                                           8, 1, 9,12,25,26,23,21,    /* child */
                                          12, 9, 2,10,26,24,22,23,    /* child */
                                          19,25,26,20, 4,13,17,16,    /* child */
                                          20,26,24,18,13, 5,14,17,    /* child */
                                          25,21,23,26,16,17,15, 7,    /* child */
                                          26,23,22,24,17,14, 6,15,    /* child */
                                          };
    PetscInt      f;
    PetscBool     signs[15];

    for (f = 0; f < 7; f++)  signs[f] = PETSC_TRUE;
    for (f = 7; f < 15; f++) signs[f] = PETSC_FALSE;
    facets[0] = hex;
    for (f = 1; f < 7; f++)  facets[f] = quad_refine_morton;
    for (f = 7; f < 15; f++) facets[f] = hex;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("hexahedron-refined-morton", 15, 27, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -24, 24, &hex_refine_morton);CHKERRQ(ierr);
  }

  { /* hilbert refined hexahedron */
    PetscPolytope facets[15];
    PetscInt      vertexOffsets[16] = {0,8, 17,26,35,44,53,62, 70,78,86,94,102,110,118,126};
    PetscInt      facetsToVertices[126] = {
                                           0, 1, 2, 3, 4, 5, 6, 7,    /* parent */
                                           0, 1, 2, 3, 8, 9,10,11,12, /* refined face */
                                           4, 5, 6, 7,13,14,15,16,17, /* refined face */
                                           0, 3, 5, 4,11,18,13,19,20, /* refined face */
                                           2, 1, 7, 6, 9,21,15,22,23, /* refined face */
                                           3, 2, 6, 5,10,22,14,18,24, /* refined face */
                                           0, 4, 7, 1,19,16,21, 8,25, /* refined face */
                                           0,19,25, 8,11,12,26,20,    /* child */
                                          11, 3,18,20,12,26,24,10,    /* child */
                                          12, 9, 2,10,26,24,22,23,    /* child */
                                          26,12, 9,23,25,21, 1, 8,    /* child */
                                          25,16, 7,21,26,23,15,17,    /* child */
                                          26,23,22,24,17,14, 6,15,    /* child */
                                          17,14,24,26,13,20,18, 5,    /* child */
                                          13,20,26,17, 4,16,25,19,    /* child */
                                          };
    PetscInt      f;
    PetscBool     signs[15];

    for (f = 0; f < 7; f++)  signs[f] = PETSC_TRUE;
    for (f = 7; f < 15; f++) signs[f] = PETSC_FALSE;
    facets[0] = hex;
    for (f = 1; f < 7; f++)  facets[f] = quad_refine_morton;
    for (f = 7; f < 15; f++) facets[f] = hex;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("hexahedron-refined-hilbert", 15, 27, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_TRUE, signs, -24, 24, &hex_refine_hilbert);CHKERRQ(ierr);
  }

  { /* wedge */
    PetscPolytope facets[5];
    PetscInt      vertexOffsets[6] = {0,3,6,10,14,18};
    PetscInt      facetsToVertices[18] = {
                                           0,1,2,
                                           3,4,5,
                                           0,3,5,1,
                                           2,4,3,0,
                                           1,5,4,2,
                                         };

    facets[0] = facets[1] = tri;
    facets[2] = facets[3] = facets[4] = quad;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("triangular-prism", 5, 6, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -6, 6, &wedge);CHKERRQ(ierr);
  }

  { /* pyramid */
    PetscPolytope facets[5];
    PetscInt      vertexOffsets[6] = {0,4,7,10,13,16};
    PetscInt      facetsToVertices[16] = {
                                           0,1,2,3,
                                           0,3,4,
                                           3,2,4,
                                           2,1,4,
                                           1,0,4,
                                         };

    facets[0] = quad;
    facets[1] = facets[2] = facets[3] = facets[4] = tri;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("pyramid", 5, 5, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -4, 4, &pyr);CHKERRQ(ierr);
  }

  { /* dodecahedron */
    PetscPolytope facets[12];
    PetscInt      f;
    PetscInt      vertexOffsets[13] = {0,5,10,15,20,25,30,35,40,45,50,55,60};
    PetscInt      facetsToVertices[60] = {
                                            0,  1,  2,  3,  4,
                                            1,  0,  5,  6,  7,
                                            2,  1,  7,  8,  9,
                                            3,  2,  9, 10, 11,
                                            4,  3, 11, 12, 13,
                                            0,  4, 13, 14,  5,
                                            6,  5, 14, 15, 16,
                                            8,  7,  6, 16, 17,
                                           10,  9,  8, 17, 18,
                                           12, 11, 10, 18, 19,
                                           14, 13, 12, 19, 15,
                                           19, 18, 17, 16, 15,
                                         };

    for (f = 0; f < 12; f++) facets[f] = pent;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("dodecahedron", 12, 20, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -60, 60, &dodec);CHKERRQ(ierr);
  }

  { /* octahedron */
    PetscPolytope facets[8];
    PetscInt      f;
    PetscInt      vertexOffsets[9] = {0,3,6,9,12,15,18,21,24};
    PetscInt      facetsToVertices[24] = {
                                           0,1,2,
                                           0,1,3,
                                           0,4,2,
                                           0,4,3,
                                           5,1,2,
                                           5,1,3,
                                           5,4,2,
                                           5,4,3,
                                         };
    PetscBool     signs[8] = {PETSC_FALSE, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE};

    for (f = 0; f < 8; f++) facets[f] = tri;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("octahedron", 8, 6, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, signs, -24, 24, &oct);CHKERRQ(ierr);
  }

  { /* rhombic dodecahedron */
    PetscPolytope facets[12];
    PetscInt      f;
    PetscInt      vertexOffsets[13] = {0,4,8,12,16,20,24,28,32,36,40,44,48};
    PetscInt      facetsToVertices[48] = {
                                            0,  1,  2,  3,
                                            1,  0,  4,  5,
                                            2,  1,  5,  6,
                                            3,  2,  7,  8,
                                            0,  3,  8,  9,
                                            4,  0,  9, 10,
                                            7,  2,  6, 11,
                                            5,  4, 10, 12,
                                           11,  6,  5, 12,
                                            8,  7, 11, 13,
                                           10,  9,  8, 13,
                                           10, 13, 11, 12,
                                         };

    for (f = 0; f < 12; f++) facets[f] = quad;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("rhombic-dodecahedron", 12, 14, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -24, 24, &rhomb);CHKERRQ(ierr);
  }

  { /* icosahedron */
    PetscPolytope facets[20];
    PetscInt      f;
    PetscInt      vertexOffsets[21] = { 0, 3, 6, 9,12,15,18,21,24,27,30,
                                          33,36,39,42,45,48,51,54,57,60};
    PetscInt      facetsToVertices[60] = {
                                            0,  1,  2,
                                            1,  0,  3,
                                            2,  1,  4,
                                            0,  2,  5,
                                            3,  0,  6,
                                            1,  3,  7,
                                            4,  1,  7,
                                            2,  4,  8,
                                            5,  2,  8,
                                            0,  5,  6,
                                            3,  6,  9,
                                            7,  3,  9,
                                            4,  7, 10,
                                            8,  4, 10,
                                            5,  8, 11,
                                            6,  5, 11,
                                            9,  6, 11,
                                            7,  9, 10,
                                            8, 10, 11,
                                           11, 10,  9,
                                         };

    for (f = 0; f < 20; f++) facets[f] = tri;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("icosahedron", 20, 12, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -60, 60, &icos);CHKERRQ(ierr);
  }

  { /* pentachoron (4-simplex) */
    PetscPolytope facets[5];
    PetscInt      f;
    PetscInt      vertexOffsets[6] = {0, 4, 8, 12, 16, 20};
    PetscInt      facetsToVertices[20] = {
                                           0,1,2,3,
                                           0,1,4,2,
                                           0,1,3,4,
                                           0,2,4,3,
                                           1,2,3,4,
                                         };

    for (f = 0; f < 5; f++) facets[f] = tet;
    ierr = PetscPolytopeInsertCheckSignsSymmetry("pentachoron", 5, 5, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -60, 60, &pchor);CHKERRQ(ierr);
  }

  { /* tesseract (4-cube) */
    PetscPolytope facets[8];
    PetscInt      f;
    PetscInt      vertexOffsets[9];
    PetscInt      facetsToVertices[64] = {
                                            0,  1,  2,  3,  4,  5,  6,  7,
                                            8,  9, 10, 11,  0,  3,  2,  1,
                                            4,  7,  6,  5, 12, 13, 14, 15,
                                            8,  0,  3, 11, 12, 13,  5,  4,
                                            1,  9, 10,  2,  7,  6, 14, 15,
                                            3,  2, 10, 11,  5, 13, 14,  6,
                                            8,  9,  1,  0, 12,  4,  7, 15,
                                            8, 11, 10,  9, 12, 15, 14, 13,
                                         };

    for (f = 0; f < 8; f++)  facets[f] = hex;
    for (f = 0; f <= 8; f++) vertexOffsets[f] = f*8;
    ierr = PetscPolytopeInsertCheckSignsSymmetry("tesseract", 8, 16, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, PETSC_FALSE, NULL, -192, 192, &tess);CHKERRQ(ierr);
  }

  { /* hyperoctahedron (4-orthoplex) */
    PetscPolytope facets[16];
    PetscInt      f;
    PetscInt      vertexOffsets[17];
    PetscInt      facetsToVertices[64] = {
                                            0,  1,  2,  3,
                                            0,  1,  2,  4,
                                            0,  1,  5,  3,
                                            0,  1,  5,  4,
                                            0,  6,  2,  3,
                                            0,  6,  2,  4,
                                            0,  6,  5,  3,
                                            0,  6,  5,  4,
                                            7,  1,  2,  3,
                                            7,  1,  2,  4,
                                            7,  1,  5,  3,
                                            7,  1,  5,  4,
                                            7,  6,  2,  3,
                                            7,  6,  2,  4,
                                            7,  6,  5,  3,
                                            7,  6,  5,  4,
                                         };
    PetscBool     signs[16] = {PETSC_TRUE,  PETSC_FALSE, PETSC_FALSE, PETSC_TRUE,  PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE,  PETSC_FALSE,
                               PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE,  PETSC_FALSE, PETSC_TRUE,  PETSC_FALSE, PETSC_FALSE, PETSC_TRUE};

    for (f = 0; f < 16; f++)  facets[f] = tet;
    for (f = 0; f <= 16; f++) vertexOffsets[f] = f*4;
    ierr = PetscPolytopeInsertCheckSignsSymmetry("hyperoctahedron", 16, 8, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, PETSC_FALSE, signs, -192, 192, &ortho);CHKERRQ(ierr);
  }

  { /* 24-cell (octaplex) */
    PetscPolytope facets[24];
    PetscInt      f, v;
    PetscInt      vertexOffsets[25];
    PetscInt      coords[96] = {
                                 -1, -1,  0,  0,
                                 -1,  1,  0,  0,
                                  1, -1,  0,  0,
                                  1,  1,  0,  0,
                                 -1,  0, -1,  0,
                                 -1,  0,  1,  0,
                                  1,  0, -1,  0,
                                  1,  0,  1,  0,
                                 -1,  0,  0, -1,
                                 -1,  0,  0,  1,
                                  1,  0,  0, -1,
                                  1,  0,  0,  1,
                                  0, -1, -1,  0,
                                  0, -1,  1,  0,
                                  0,  1, -1,  0,
                                  0,  1,  1,  0,
                                  0, -1,  0, -1,
                                  0, -1,  0,  1,
                                  0,  1,  0, -1,
                                  0,  1,  0,  1,
                                  0,  0, -1, -1,
                                  0,  0, -1,  1,
                                  0,  0,  1, -1,
                                  0,  0,  1,  1,
                               };
    PetscInt     vertices[144];
    PetscInt     vertid[24], vcount, fcount;

    for (f = 0; f < 24; f++)  facets[f] = oct;
    for (f = 0; f <= 24; f++) vertexOffsets[f] = f * 6;
    for (v = 0; v < 144; v++) vertices[v] = -1;
    for (v = 0; v < 24; v++) vertid[v] = -1;
    for (vcount = 0, fcount = 0, v = 0; v < 24; v++) {
      PetscInt w;

      for (w = v+1; w < 24; w++) {
        PetscInt diff, j, k, x[3], xcount;

        for (diff = 0, j = 0; j < 4; j++) diff += PetscSqr(coords[4 * w + j] - coords[4 * v + j]);
        if (diff != 2) continue;
        for (xcount = 0, k = 0; k < 24; k++) {
          for (diff = 0, j = 0; j < 4; j++) diff += PetscSqr(coords[4 * k + j] - coords[4 * v + j]);
          if (diff != 2) continue;
          for (diff = 0, j = 0; j < 4; j++) diff += PetscSqr(coords[4 * k + j] - coords[4 * w + j]);
          if (diff != 2) continue;
          if (xcount == 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Too many faces around an edge");
          x[xcount++] = k;
        }
        if (xcount != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Too few faces around an edge");
        for (k = 0; k < 3; k++) {
          PetscInt z, y, l;
          PetscReal mid[4];
          PetscReal dispv[4];
          PetscReal dispw[4];
          PetscInt  oppvc[4];
          PetscInt  oppwc[4];
          PetscInt  oppv, oppw;
          PetscInt  octohedron[6];
          PetscInt  mat[4][4];
          PetscInt  det;

          z = x[k];
          y = x[(k+1)%3];
          if (y < z) {
            PetscInt swap = y;
            y = z;
            z = swap;
          }
          if (z < w) continue;
          for (diff = 0, j = 0; j < 4; j++) diff += PetscSqr(coords[4 * z + j] - coords[4 * y + j]);
          if (diff != 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect geometry");
          for (j = 0; j < 4; j++) mid[j] = coords[4 * z + j] + ((PetscReal) coords[4 * y + j] - coords[4 * z + j]) / 2.;
          for (j = 0; j < 4; j++) dispv[j] = ((PetscReal) coords[4 * v + j] - mid[j]);
          for (j = 0; j < 4; j++) dispw[j] = ((PetscReal) coords[4 * w + j] - mid[j]);
          for (j = 0; j < 4; j++) oppvc[j] = mid[j] - dispv[j];
          for (j = 0; j < 4; j++) oppwc[j] = mid[j] - dispw[j];
          for (l = 0; l < 24; l++) {
            for (j = 0; j < 4; j++) if (oppvc[j] != coords[4 * l + j]) break;
            if (j == 4) {
              oppv = l;
              break;
            }
          }
          for (l = 0; l < 24; l++) {
            for (j = 0; j < 4; j++) if (oppwc[j] != coords[4 * l + j]) break;
            if (j == 4) {
              oppw = l;
              break;
            }
          }
          if (oppv < v || oppw < w) continue;
          octohedron[0] = v;
          octohedron[1] = w;
          octohedron[2] = z;
          octohedron[3] = y;
          octohedron[4] = oppw;
          octohedron[5] = oppv;
          for (l = 0; l < 4; l++) for (j = 0; j < 4; j++) mat[l][j] = coords[octohedron[l] * 4 + j];
          for (det = 0, l = 0; l < 24; l++) {
            PetscInt perm[4] = {0, 1, 2, 3};
            PetscInt odd = 0, swap, sign;
            PetscInt p = l;

            j = p % 4;
            p = p / 4;
            if (j != 0) odd ^= 1;
            swap = perm[j];
            perm[j] = perm[0];
            perm[0] = swap;

            j = p % 3;
            p = p / 3;
            if (j != 0) odd ^= 1;
            swap = perm[j+1];
            perm[j+1] = perm[1];
            perm[1] = swap;

            j = p % 2;
            p = p / 2;
            if (j != 0) odd ^= 1;
            swap = perm[j+1];
            perm[j+1] = perm[1];
            perm[1] = swap;

            sign = (odd) ? -1 : 1;
            det += sign * mat[0][perm[0]] * mat[1][perm[1]] * mat[2][perm[2]] * mat[3][perm[3]];
          }
          if (det == 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "degenerate tetrahedron");
          if (det < 0) {
            PetscInt swap;

            swap = octohedron[2];
            octohedron[2] = octohedron[3];
            octohedron[3] = swap;
          }
          for (l = 0; l < 6; l++) {
            if (vertid[octohedron[l]] == -1) vertid[octohedron[l]] = vcount++;
            vertices[6 * fcount + l] = vertid[octohedron[l]];
          }
          fcount++;
        }
      }
    }
    ierr = PetscPolytopeInsertCheckSignsSymmetry("octoplex", 24, 24, facets, vertexOffsets, vertices, PETSC_FALSE, PETSC_FALSE, NULL, -576, 576, &oplex);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
