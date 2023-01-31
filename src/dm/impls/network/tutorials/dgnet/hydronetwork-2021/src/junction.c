#include "wash.h"

/* Subroutines for Junction                               */
/* -------------------------------------------------------*/

/*
    JunctionCreateJacobian - Create Jacobian matrices for a vertex.

    Collective on Pipe

    Input Parameter:
+   dm - the DMNetwork object
.   v - vertex point
-   Jin - Jacobian patterns created by JunctionCreateJacobianSample() for reuse

    Output Parameter:
.   J  - array of Jacobian matrices (see dmnetworkimpl.h)

    Level: beginner
*/
PetscErrorCode JunctionCreateJacobian(DM dm, PetscInt v, Mat *Jin, Mat *J[])
{
  PetscErrorCode  ierr;
  Mat            *Jv;
  PetscInt        nedges, e, i, M, N, *rows, *cols;
  PetscBool       isSelf;
  const PetscInt *edges, *cone;
  PetscScalar    *zeros;

  PetscFunctionBegin;
  /* Get arrary size of Jv */
  ierr = DMNetworkGetSupportingEdges(dm, v, &nedges, &edges);
  CHKERRQ(ierr);
  PetscCheck(nedges, PETSC_COMM_SELF, 1, "%d vertex, nedges %d\n", v, nedges);

  /* two Jacobians for each connected edge: J(v,e) and J(v,vc); adding J(v,v), total 2*nedges+1 Jacobians */
  ierr = PetscCalloc1(2 * nedges + 1, &Jv);
  CHKERRQ(ierr);

  /* Create dense zero block for this vertex: J[0] = Jacobian(v,v) */
  ierr = DMNetworkGetComponent(dm, v, ALL_COMPONENTS, NULL, NULL, &M);
  CHKERRQ(ierr);
  PetscCheck(M == 2, PETSC_COMM_SELF, 1, "M != 2");

  ierr = PetscMalloc3(M, &rows, M, &cols, M * M, &zeros);
  CHKERRQ(ierr);
  ierr = PetscMemzero(zeros, M * M * sizeof(PetscScalar));
  CHKERRQ(ierr);
  for (i = 0; i < M; i++) rows[i] = i;

  for (e = 0; e < nedges; e++) {
    /* create Jv[2*e+1] = Jacobian(v,e), e: supporting edge */
    ierr = DMNetworkGetConnectedVertices(dm, edges[e], &cone);
    CHKERRQ(ierr);
    isSelf = (v == cone[0]) ? PETSC_TRUE : PETSC_FALSE;

    if (Jin) {
      if (isSelf) {
        Jv[2 * e + 1] = Jin[0];
      } else {
        Jv[2 * e + 1] = Jin[1];
      }
      Jv[2 * e + 2] = Jin[2];
      ierr          = PetscObjectReference((PetscObject)(Jv[2 * e + 1]));
      CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)(Jv[2 * e + 2]));
      CHKERRQ(ierr);
    } else {
      /* create J(v,e) */
      ierr = MatCreate(PETSC_COMM_SELF, &Jv[2 * e + 1]);
      CHKERRQ(ierr);
      ierr = DMNetworkGetComponent(dm, edges[e], ALL_COMPONENTS, NULL, NULL, &N);
      CHKERRQ(ierr);
      ierr = MatSetSizes(Jv[2 * e + 1], PETSC_DECIDE, PETSC_DECIDE, M, N);
      CHKERRQ(ierr);
      ierr = MatSetFromOptions(Jv[2 * e + 1]);
      CHKERRQ(ierr);
      //ierr = MatSetOption(Jv[2*e+1],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr); '-mat_view draw' crash!
      ierr = MatSeqAIJSetPreallocation(Jv[2 * e + 1], 2, NULL);
      CHKERRQ(ierr);
      if (N) {
        if (isSelf) { /* coupling at upstream */
          for (i = 0; i < 2; i++) cols[i] = i;
        } else { /* coupling at downstream */
          cols[0] = N - 2;
          cols[1] = N - 1;
        }
        ierr = MatSetValues(Jv[2 * e + 1], 2, rows, 2, cols, zeros, INSERT_VALUES);
        CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(Jv[2 * e + 1], MAT_FINAL_ASSEMBLY);
      CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Jv[2 * e + 1], MAT_FINAL_ASSEMBLY);
      CHKERRQ(ierr);

      /* create Jv[2*e+2] = Jacobian(v,vc), vc: connected vertex.
       In WashNetwork, v and vc are not connected, thus Jacobian(v,vc) is empty */
      ierr = MatCreate(PETSC_COMM_SELF, &Jv[2 * e + 2]);
      CHKERRQ(ierr);
      ierr = MatSetSizes(Jv[2 * e + 2], PETSC_DECIDE, PETSC_DECIDE, M, M);
      CHKERRQ(ierr); /* empty matrix, sizes can be arbitrary */
      ierr = MatSetFromOptions(Jv[2 * e + 2]);
      CHKERRQ(ierr);
      //ierr = MatSetOption(Jv[2*e+2],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(Jv[2 * e + 2], 1, NULL);
      CHKERRQ(ierr);
      ierr = MatAssemblyBegin(Jv[2 * e + 2], MAT_FINAL_ASSEMBLY);
      CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Jv[2 * e + 2], MAT_FINAL_ASSEMBLY);
      CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(rows, cols, zeros);
  CHKERRQ(ierr);

  *J = Jv;
  PetscFunctionReturn(0);
}

PetscErrorCode JunctionDestroyJacobian(DM dm, PetscInt v, Junction junc)
{
  PetscErrorCode  ierr;
  Mat            *Jv = junc->jacobian;
  const PetscInt *edges;
  PetscInt        nedges, e;

  PetscFunctionBegin;
  if (!Jv) PetscFunctionReturn(0);

  ierr = DMNetworkGetSupportingEdges(dm, v, &nedges, &edges);
  CHKERRQ(ierr);
  for (e = 0; e < nedges; e++) {
    ierr = MatDestroy(&Jv[2 * e + 1]);
    CHKERRQ(ierr);
    ierr = MatDestroy(&Jv[2 * e + 2]);
    CHKERRQ(ierr);
  }
  ierr = PetscFree(Jv);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
