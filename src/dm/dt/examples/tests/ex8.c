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

int main(int argc, char **argv)
{
  PetscPolytope  null, no_point, vertex, edge, triangle, quad, pent;
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

    ierr = PetscPolytopeInsertCheck("vertex", 1, 0, &null, vertexOffsets, NULL, PETSC_FALSE, &vertex);CHKERRQ(ierr);
    ierr = PetscPolytopeInsertCheck("vertex-inverted", 1, 0, &null, vertexOffsets, NULL, PETSC_TRUE, &inv_vertex);CHKERRQ(ierr);
    if (inv_vertex == vertex) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverted vertex the same as vertex");CHKERRQ(ierr);
    ierr = PetscPolytopeGetOrientationRange(vertex, &oStart, &oEnd);CHKERRQ(ierr);
    if (oStart != 0 || oEnd != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of vertex");
  }

  {
    PetscPolytope facets[2];
    PetscInt      vertexOffsets[3] = {0,1,2};
    PetscInt      facetsToVertices[2] = {0,1};
    const PetscBool *facetsInward;
    PetscPolytope inv_edge;

    facets[0] = vertex;
    facets[1] = vertex;

    ierr = PetscPolytopeInsertCheck("edge", 2, 2, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, &edge);CHKERRQ(ierr);
    ierr = PetscPolytopeInsertCheck("edge-inverted", 2, 2, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &inv_edge);CHKERRQ(ierr);
    if (inv_edge == edge) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverted edge the same as edge");
    ierr = PetscPolytopeGetData(edge, NULL, NULL, NULL, NULL, NULL, &facetsInward);CHKERRQ(ierr);
    if (facetsInward[1] != PETSC_FALSE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "edge does not have oppositely signed vertices");
    ierr = PetscPolytopeGetData(inv_edge, NULL, NULL, NULL, NULL, NULL, &facetsInward);CHKERRQ(ierr);
    if (facetsInward[1] != PETSC_TRUE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "edge does not have oppositely signed vertices");
    ierr = PetscPolytopeGetOrientationRange(edge, &oStart, &oEnd);CHKERRQ(ierr);
    if (oStart != -1 || oEnd != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of edge");
    ierr = PetscPolytopeGetOrientationRange(inv_edge, &oStart, &oEnd);CHKERRQ(ierr);
    if (oStart != -1 || oEnd != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of edge");
  }

#if 0
  { /* try and fail to create a mixed dimension polytope */
    PetscPolytope facets[2];
    PetscInt      vertexOffsets[3] = {0,1,3};
    PetscInt      facetsToVertices[2] = {0,1,2};
    const PetscBool *facetsInward;
    PetscPolytope inv_edge;

    facets[0] = vertex;
    facets[1] = edge;

    testerr = PetscPolytopeInsert("mixed-triangle", 2, 3, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, &edge);
    if (testerr != PETSC_ERR_ARG_WRONG) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not catch mixed dimension polytope");
  }
#endif

  { /* cyclic triangle */
    PetscPolytope facets[3];
    PetscInt      vertexOffsets[4] = {0,2,4,6};
    PetscInt      facetsToVertices[6] = {0,1,1,2,2,0};
    const PetscBool *facetsInward;

    facets[0] = edge;
    facets[1] = edge;
    facets[2] = edge;

    ierr = PetscPolytopeInsertCheck("triangle", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &triangle);CHKERRQ(ierr);
    ierr = PetscPolytopeGetData(triangle, NULL, NULL, NULL, NULL, NULL, &facetsInward);CHKERRQ(ierr);
    if (facetsInward[1] != PETSC_FALSE || facetsInward[2] != PETSC_FALSE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "triangle does not have outward edges");
    ierr = PetscPolytopeGetOrientationRange(triangle, &oStart, &oEnd);CHKERRQ(ierr);
    if (oStart != -3 || oEnd != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of triangle");
  }

  { /* non-cyclic triangle */
    PetscPolytope facets[3], noncyctri;
    PetscInt      vertexOffsets[4] = {0,2,4,6};
    PetscInt      facetsToVertices[6] = {0,1,0,2,1,2};
    const PetscBool *facetsInward;

    facets[0] = edge;
    facets[1] = edge;
    facets[2] = edge;

    ierr = PetscPolytopeInsertCheck("non-cyclic-triangle", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &noncyctri);CHKERRQ(ierr);
    ierr = PetscPolytopeGetData(noncyctri, NULL, NULL, NULL, NULL, NULL, &facetsInward);CHKERRQ(ierr);
    if (facetsInward[1] != PETSC_TRUE || facetsInward[2] != PETSC_FALSE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "non-cyclic triangle does not have correct outward/inward edges");
    ierr = PetscPolytopeGetOrientationRange(noncyctri, &oStart, &oEnd);CHKERRQ(ierr);
    if (oStart != -3 || oEnd != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of non-cyclic triangle");
  }

  { /* cyclic quadrilateral */
    PetscPolytope facets[4];
    PetscInt      vertexOffsets[5] = {0,2,4,6,8};
    PetscInt      facetsToVertices[8] = {0,1,1,2,2,3,3,0};
    const PetscBool *facetsInward;

    facets[0] = edge;
    facets[1] = edge;
    facets[2] = edge;
    facets[3] = edge;

    ierr = PetscPolytopeInsertCheck("quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &quad);CHKERRQ(ierr);
    ierr = PetscPolytopeGetData(quad, NULL, NULL, NULL, NULL, NULL, &facetsInward);CHKERRQ(ierr);
    if (facetsInward[1] != PETSC_FALSE || facetsInward[2] != PETSC_FALSE || facetsInward[3] != PETSC_FALSE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "quadrilateral does not have outward edges");
    ierr = PetscPolytopeGetOrientationRange(quad, &oStart, &oEnd);CHKERRQ(ierr);
    if (oStart != -4 || oEnd != 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of quadrilateral");
  }

  { /* non-cyclic quadrilateral */
    PetscPolytope facets[4], noncycquad;
    PetscInt      vertexOffsets[5] = {0,2,4,6,8};
    PetscInt      facetsToVertices[8] = {0,1,2,3,0,2,1,3};
    const PetscBool *facetsInward;

    facets[0] = edge;
    facets[1] = edge;
    facets[2] = edge;
    facets[3] = edge;

    ierr = PetscPolytopeInsertCheck("non-cyclic-quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &noncycquad);CHKERRQ(ierr);
    ierr = PetscPolytopeGetData(noncycquad, NULL, NULL, NULL, NULL, NULL, &facetsInward);CHKERRQ(ierr);
    if (facetsInward[1] != PETSC_TRUE || facetsInward[2] != PETSC_TRUE || facetsInward[3] != PETSC_FALSE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "non-cyclic quadrilateral does not have correct outward/inward edges");
    ierr = PetscPolytopeGetOrientationRange(noncycquad, &oStart, &oEnd);CHKERRQ(ierr);
    if (oStart != -4 || oEnd != 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect symmetry group of non-cyclic quadrilateral");
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
