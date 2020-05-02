#include "petscpc.h"
static char help[] = "Test PCFieldSplit in Distributed Gauss-Seidel mode, with very simple 1D system\n\n";

#include <petscdmstag.h>
#include <petscksp.h>

static PetscErrorCode CreateSystem(DM dm,Mat *p_A,Vec *p_b)
{
  PetscInt     N,n,start;
  Mat          A;
  Vec          b, constant_pressure;
  MatNullSpace A_nullspace;

  PetscFunctionBeginUser;

  /*
  A very simple system which (counterintuitively) puts pressures on the
  vertices and velocities on the elements. With implicit zero velocity
  ghost nodes, then, this provides a system for which the commutator
  AG - GDG = 0 (everywhere, including on the boundaries) which is useful
  for testing as Richardson iteration with a DGS preconditioner with
  exact subsolves, specifying the correct nullspace for the auxiliary operator,
  Ap = DG) will converge in one iteration.
  */

  PetscCall(DMSetMatrixPreallocateOnly(dm,PETSC_TRUE));
  PetscCall(DMCreateMatrix(dm,p_A));
  A = *p_A;

  PetscCall(DMCreateGlobalVector(dm,p_b));
  b = *p_b;

  PetscCall(VecDuplicate(b,&constant_pressure));

  PetscCall(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N,NULL,NULL));
  for (PetscInt e = start; e<start+n; ++e) {

    // Velocity-Velocity (on element) coupling
    if (e == 0) {
      DMStagStencil row, col[2];
      PetscScalar   val[2];

      row.c = 0;
      row.loc = DMSTAG_ELEMENT;
      row.i = e;
      col[0].c = 0;
      col[0].loc = DMSTAG_ELEMENT;
      col[0].i = e;
      val[0] = 2.0;
      col[1].c = 0;
      col[1].loc = DMSTAG_ELEMENT;
      val[1] = -1.0;
      col[1].i = e+1;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,2,col,val,INSERT_VALUES));
    } else if (e == N-1) {
      DMStagStencil row, col[2];
      PetscScalar   val[2];

      row.c = 0;
      row.loc = DMSTAG_ELEMENT;
      row.i = e;
      col[0].c = 0;
      col[0].loc = DMSTAG_ELEMENT;
      col[0].i = e-1;
      val[0] = -1.0;
      col[1].c = 0;
      col[1].loc = DMSTAG_ELEMENT;
      val[1] = 2.0;
      col[1].i = e;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,2,col,val,INSERT_VALUES));
    } else {
      DMStagStencil row, col[3];
      PetscScalar   val[3];

      row.c = 0;
      row.loc = DMSTAG_ELEMENT;
      row.i = e;
      col[0].c = 0;
      col[0].loc = DMSTAG_ELEMENT;
      col[0].i = e-1;
      val[0] = -1.0;
      col[1].c = 0;
      col[1].loc = DMSTAG_ELEMENT;
      val[1] = 2.0;
      col[1].i = e;
      col[2].c = 0;
      col[2].loc = DMSTAG_ELEMENT;
      val[2] = -1.0;
      col[2].i = e+1;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,3,col,val,INSERT_VALUES));
    }


    // Velocity (on elements) to pressure (on vertices) coupling
    // Discrete gradient
    {
      DMStagStencil row, col[2];
      PetscScalar   val[2];

      row.c = 0;
      row.loc = DMSTAG_ELEMENT;
      row.i = e;
      col[0].c = 0;
      col[0].loc = DMSTAG_LEFT;
      col[0].i = e;
      val[0] = -1.0;
      col[1].c = 0;
      col[1].loc = DMSTAG_RIGHT;
      val[1] = 1.0;
      col[1].i = e;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,2,col,val,INSERT_VALUES));
    }

    // Pressure (on vertices) to velocity (on elements) coupling
    // Discrete Divergence
    if (e == 0) {
      DMStagStencil row, col;
      PetscScalar   val;

      row.loc = DMSTAG_LEFT;
      row.i = 0;
      row.c = 0;
      col.loc = DMSTAG_ELEMENT;
      col.c = 0;
      col.i = 0;
      val = -1.0;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,1,&col,&val,INSERT_VALUES));
    } else if (e == N-1) {
      DMStagStencil row, col;
      PetscScalar   val;

      row.loc = DMSTAG_RIGHT;
      row.i = N-1;
      row.c = 0;
      col.loc = DMSTAG_ELEMENT;
      col.c = 0;
      col.i = N-1;
      val = 1.0;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,1,&col,&val,INSERT_VALUES));
    }

    if(e < N-1) {
      DMStagStencil row, col[2];
      PetscScalar   val[2];

      row.loc = DMSTAG_RIGHT;
      row.c = 0;
      row.i = e;
      col[0].c = 0;
      col[0].loc = DMSTAG_ELEMENT;
      col[0].i = e;
      val[0] = 1.0;
      col[1].c = 0;
      col[1].loc = DMSTAG_ELEMENT;
      col[1].i = e+1;
      val[1] = -1.0;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,2,col,val,INSERT_VALUES));
    }

    // Explicit zero pressure-pressure couplings (on vertices)
    {
      DMStagStencil row;
      PetscScalar   val;

      row.loc = DMSTAG_LEFT;
      row.i = e;
      row.c = 0;
      val = 0.0;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,1,&row,&val,INSERT_VALUES));
    }
    if (e == N-1) {
      DMStagStencil row;
      PetscScalar   val;

      row.loc = DMSTAG_RIGHT;
      row.i = e;
      row.c = 0;
      val = 0.0;
      PetscCall(DMStagMatSetValuesStencil(dm,A,1,&row,1,&row,&val,INSERT_VALUES));
    }

    // Add entries to RHS vector, forcing on velocity (element) DOF
    {
      DMStagStencil row;
      PetscScalar val;

      row.loc = DMSTAG_ELEMENT;
      row.i = e;
      val = 1.0;
      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &val, INSERT_VALUES));
    }

    // Add entry to nullspace (constant pressure, on vertices) vector
    {
      DMStagStencil row;
      PetscScalar val;

      row.loc = DMSTAG_LEFT;
      row.i = e;
      val = 1.0;
      PetscCall(DMStagVecSetValuesStencil(dm, constant_pressure, 1, &row, &val, INSERT_VALUES));
    }
    if (e == N-1) {
      DMStagStencil row;
      PetscScalar val;

      row.loc = DMSTAG_RIGHT;
      row.i = e;
      val = 1.0;
      PetscCall(DMStagVecSetValuesStencil(dm, constant_pressure, 1, &row, &val, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(VecAssemblyBegin(constant_pressure));
  PetscCall(VecAssemblyEnd(constant_pressure));
  PetscCall(VecNormalize(constant_pressure,NULL));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm),PETSC_FALSE,1,&constant_pressure,&A_nullspace));
  PetscCall(VecDestroy(&constant_pressure));
  PetscCall(MatSetNullSpace(A,A_nullspace));
  PetscCall(MatNullSpaceDestroy(&A_nullspace));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DM       dm;
  Mat      A;
  Vec      x,b;
  KSP      ksp;
  PC       pc;
  PetscInt n_fields;
  IS       *fields;
  char     **names;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));


  PetscCall(CreateSystem(dm,&A,&b));

  // Get the elements-only split (velocity),
  PetscCall(DMCreateFieldDecomposition(dm,&n_fields,&names,&fields,NULL));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm),&ksp));
  PetscCall(KSPSetType(ksp,KSPFGMRES));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCFIELDSPLIT));
  PetscCall(PCFieldSplitSetType(pc,PC_COMPOSITE_DGS));


  PetscCall(PCFieldSplitSetIS(pc,names[1],fields[1])); // FIXME gross -  we had to know that this was elements (could have checked the name but yuck)


  PetscCall(PCSetOperators(pc,A,A));
  PetscCall(PCSetDM(pc,dm));


  // FIXME: Kludge for now: set up the PC, then pull out the auxiliar operator and say it has
  // a nullspace of constants...
  {
    Mat          Ap;
    MatNullSpace nullspace;

    PetscCall(PCSetUp(pc));
    PetscCall(PCFieldSplitDGSGetAuxiliaryOperator(pc, &Ap));
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm),PETSC_TRUE,0,NULL,&nullspace));
    PetscCall(MatSetNullSpace(Ap,nullspace));
    PetscCall(MatNullSpaceDestroy(&nullspace));
  }

  /* Solve */
  PetscCall(KSPSetFromOptions(ksp)); // FIXME we already called PCSetUp so this won't work as expected
  PetscCall(VecDuplicate(b,&x));
  PetscCall(KSPSolve(ksp,b,x));

  /* Clean up */
  for (PetscInt i=0; i<n_fields; ++i) {
    PetscCall(ISDestroy(&fields[i]));
    PetscCall(PetscFree(names[i]));
  }
  PetscCall(PetscFree(fields));
  PetscCall(PetscFree(names));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 1
      args: -ksp_converged_reason -fieldsplit_dgs_aux_ksp_converged_reason -fieldsplit_dgs_aux_pc_type jacobi -stag_grid_x 11

TEST*/
