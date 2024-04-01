#include <petsc/private/dmpleximpl.h>  /*I      "petscdmplex.h"   I*/

// Cells are cylindrical annuli. We can locate points by converting to (r, z) and locating in 2D.
// TODO: For now, just pass in the 2D DM. We should eventually wrap it in a DM for the cylindrical mesh
PetscErrorCode DMLocatePoints_Plex_Cylindrical(DM dm, Vec v, DMPointLocationType ltype, PetscSF cellSF)
{
  Vec                v2d;
  VecType            vtype;
  const PetscScalar *a;
  PetscScalar       *a2d;
  PetscInt           bs, n;

  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(v, &bs));
  PetscCheck(bs == 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Cylindrical meshes take 3D coordinates, not %" PetscInt_FMT, bs);
  PetscCall(VecCreate(PetscObjectComm((PetscObject)v), &v2d));
  PetscCall(VecGetType(v, &vtype));
  PetscCall(VecSetType(v2d, vtype));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecSetLocalSize(v2d, (n / 3) * 2));
  PetscCall(VecGetArrayRead(v, &a));
  PetscCall(VecGetArray(v2d, &a2d));
  PetscCall(VecRestoreArrayRead(v, &a));
  for (PetscInt i = 0; i < n / 3; ++i) {
    a2d[i * 2 + 0] = PetscSqrtReal(PetscSqr(PetscRealPart(a[i * 3 + 0])) + PetscSqr(PetscRealPart(a[i * 3 + 1])));
    a2d[i * 2 + 1] = a[i * 3 + 2];
  }
  PetscCall(VecRestoreArray(v2d, &a2d));
  PetscCall(DMLocatePoints_Plex(dm, v2d, ltype, cellSF));
  PetscCall(VecDestroy(&v2d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMCylindricalGetLineCellIntersection - Get the intersection of a line with the cylindrical cell

  Not collective

  Input Parameters:
+ dm - the DM
. c  - the mesh point
. p0 - a point on the line
- p1 - a second point on the line

  Output Parameters:
. pos       - `PETSC_TRUE` is the cell is on the positive z side of the line, `PETSC_FALSE` on the negative z side
+ Nint      - the number of intersection points, in [0, 2]
- intPoints - the coordinates of the intersection points, should be length at least 6

  Note: The `pos` argument is only meaningful if the number of intersections is 0. The algorithmic idea comes from https://github.com/chrisk314/tet-plane-intersection.

  Level: developer

.seealso:
*/
PetscErrorCode DMCylindricalGetLineCellIntersection(DM dm, PetscInt c, PetscReal p0[], PetscReal p1[], PetscBool *pos, PetscInt *Nint, PetscReal intPoints[])
{
  PetscReal p[2], q[2], normal[2], intPoints2D[4 * 2];

  PetscFunctionBegin;
  // Project line into a plane containing the z-axis by setting y = 0, or puting it in the x-z plane
  //   In this plane, the cell cross-section is exactly our 2D mesh
  //   Order points so that p1 has smaller r
  p[0]      = p0[0] < p1[0] ? p0[0] : p1[0];
  p[1]      = p0[0] < p1[0] ? p0[2] : p1[2];
  q[0]      = p0[0] < p1[0] ? p1[0] : p0[0];
  q[1]      = p0[0] < p1[0] ? p1[2] : p0[2];
  // Normal is in \hat\phi direction
  normal[0] = q[1] - p[1];
  normal[1] = -(q[0] - p[0]);

  PetscCall(DMPlexGetPlaneCellIntersection_Internal(dm, c, p, normal, pos, Nint, intPoints2D));
  // Convert intersection points
  for (PetscInt i = 0; i < *Nint; ++i) {
    const PetscReal xSlope = (p1[0] - p0[0]) / (p1[2] - p0[2]);
    const PetscReal ySlope = (p1[1] - p0[1]) / (p1[2] - p0[2]);
    const PetscReal dz     = intPoints2D[i * 2 + 1] - p0[2];

    intPoints[i * 3 + 0] = xSlope * dz;
    intPoints[i * 3 + 1] = ySlope * dz;
    intPoints[i * 3 + 2] = intPoints2D[i * 2 + 1];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
