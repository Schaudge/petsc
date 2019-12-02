static char help[] = "Tests PetscPolytope.\n\n";

#include <petscdt.h>

static PetscErrorCode PetscPolytopeInsertCheck(const char name[], PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscPolytope *polytope)
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
  ierr = PetscPolytopeInsert(name, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &p);CHKERRQ(ierr);
  ierr = PetscSNPrintf(namedup, 256, "%s-dup", name);CHKERRQ(ierr);
  ierr = PetscPolytopeInsert(namedup, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &pagain);CHKERRQ(ierr);
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
                                                            const PetscBool signs[], PetscInt oStart, PetscInt oEnd, PetscPolytope *polytope)
{
  PetscInt f, coStart, coEnd, o, w, wo, v;
  const PetscBool *facetsInward;
  PetscInt       *ov, *wv, *wov;
  PetscInt       *of, *wf, *wof;
  PetscInt       *ofo, *wfo, *wofo;
  PetscPolytope  p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopeInsertCheck(name, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &p);CHKERRQ(ierr);
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
                 tet, noncyctet, hex, pent, wedge, pyr, dodec, oct, rhomb, icos, pchor, tess, ortho, oplex;
  PetscInt       oStart, oEnd;
  PetscErrorCode ierr, testerr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscPolytopeInsertCheck("null", 0, 0, NULL, NULL, NULL, PETSC_FALSE, &null);CHKERRQ(ierr);
  ierr = PetscPolytopeGetByName("not-a-point", &no_point);CHKERRQ(ierr);
  if (no_point != PETSCPOLYTOPE_NONE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unrecognized name did not return PETSCPOLYTOPE_NONE\n");
  ierr = PetscPolytopeGetOrientationRange(null, &oStart, &oEnd);CHKERRQ(ierr);
  if (oStart != 0 || oEnd != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of null point");

  {
    PetscInt vertexOffsets[2] = {0,0};
    PetscPolytope inv_vertex;
    PetscBool signs[1] = {PETSC_FALSE};
    PetscBool inv_signs[1] = {PETSC_TRUE};

    ierr = PetscPolytopeInsertCheckSignsSymmetry("vertex", 1, 0, &null, vertexOffsets, NULL, PETSC_FALSE, signs, 0, 1, &vertex);CHKERRQ(ierr);
    ierr = PetscPolytopeInsertCheckSignsSymmetry("vertex-inverted", 1, 0, &null, vertexOffsets, NULL, PETSC_TRUE, inv_signs, 0, 1, &inv_vertex);CHKERRQ(ierr);
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("edge", 2, 2, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, signs, -1, 1, &edge);CHKERRQ(ierr);
    ierr = PetscPolytopeInsertCheckSignsSymmetry("edge-inverted", 2, 2, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, inv_signs, -1, 1, &inv_edge);CHKERRQ(ierr);
    if (inv_edge == edge) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverted edge the same as edge");
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("triangle", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -3, 3, &tri);CHKERRQ(ierr);
  }

  { /* non-cyclic triangle */
    PetscPolytope facets[3];
    PetscInt      vertexOffsets[4] = {0,2,4,6};
    PetscInt      facetsToVertices[6] = {0,1, 0,2, 1,2};
    PetscBool     signs[3] = {PETSC_FALSE, PETSC_TRUE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("non-cyclic-triangle", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, signs, -3, 3, &noncyctri);CHKERRQ(ierr);
  }

  { /* cyclic quadrilateral */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,2,4,6,8};
    PetscInt      facetsToVertices[8] = {0,1, 1,2, 2,3, 3,0};

    facets[0] = edge;
    facets[1] = edge;
    facets[2] = edge;
    facets[3] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -4, 4, &quad);CHKERRQ(ierr);
  }

  { /* non-cyclic quadrilateral */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,2,4,6,8};
    PetscInt      facetsToVertices[8] = {0,1, 2,3, 0,2, 1,3};
    PetscBool     signs[4] = {PETSC_FALSE, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = facets[3] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("non-cyclic-quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, signs, -4, 4, &noncycquad);CHKERRQ(ierr);
  }

  { /* cyclic pentagon */
    PetscPolytope facets[5];
    PetscInt      vertexOffsets[6] = {0,2,4,6,8,10};
    PetscInt      facetsToVertices[10] = {0,1, 1,2, 2,3, 3,4, 4,0};

    facets[0] = facets[1] = facets[2] = facets[3] = facets[4] = edge;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("pentagon", 5, 5, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -5, 5, &pent);CHKERRQ(ierr);
  }

  { /* tetrahedron */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,3,6,9,12};
    PetscInt      facetsToVertices[12] = {0,1,2, 0,3,1, 0,2,3, 2,1,3};

    facets[0] = facets[1] = facets[2] = facets[3] = tri;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("tetrahedron", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -12, 12, &tet);CHKERRQ(ierr);
  }

  { /* non-cyclic tetrahedron */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,3,6,9,12};
    PetscInt      facetsToVertices[12] = {0,1,2, 0,1,3, 0,2,3, 1,2,3};
    PetscBool     signs[4] = {PETSC_TRUE, PETSC_FALSE, PETSC_TRUE, PETSC_FALSE};

    facets[0] = facets[1] = facets[2] = facets[3] = noncyctri;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("non-cyclic tetrahedron", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, signs, -12, 12, &noncyctet);CHKERRQ(ierr);
  }

  { /* hexahedron */
    PetscPolytope facets[6];
    PetscInt      vertexOffsets[7] = {0,4,8,12,16,20,24};
    PetscInt      facetsToVertices[24] = {0,1,2,3, 4,5,6,7, 0,3,5,4, 2,1,7,6, 3,2,6,5, 0,4,7,1};

    facets[0] = facets[1] = facets[2] = facets[3] = facets[4] = facets[5] = quad;

    ierr = PetscPolytopeInsertCheckSignsSymmetry("hexahedron", 6, 8, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -24, 24, &hex);CHKERRQ(ierr);
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("triangular-prism", 5, 6, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -6, 6, &wedge);CHKERRQ(ierr);
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("pyramid", 5, 5, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -4, 4, &pyr);CHKERRQ(ierr);
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("dodecahedron", 12, 20, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -60, 60, &dodec);CHKERRQ(ierr);
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("octahedron", 8, 6, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, signs, -24, 24, &oct);CHKERRQ(ierr);
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("rhombic-dodecahedron", 12, 14, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -24, 24, &rhomb);CHKERRQ(ierr);
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

    ierr = PetscPolytopeInsertCheckSignsSymmetry("icosahedron", 20, 12, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -60, 60, &icos);CHKERRQ(ierr);
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
    ierr = PetscPolytopeInsertCheckSignsSymmetry("pentachoron", 5, 5, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -60, 60, &pchor);CHKERRQ(ierr);
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
    ierr = PetscPolytopeInsertCheckSignsSymmetry("tesseract", 8, 16, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, NULL, -192, 192, &tess);CHKERRQ(ierr);
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
    ierr = PetscPolytopeInsertCheckSignsSymmetry("hyperoctahedron", 16, 8, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, signs, -192, 192, &ortho);CHKERRQ(ierr);
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
    ierr = PetscPolytopeInsertCheckSignsSymmetry("octoplex", 24, 24, facets, vertexOffsets, vertices, PETSC_FALSE, NULL, -576, 576, &ortho);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
