const char help[] = "A simple boundary value problem meant to demonstrate basic finite element functionality.";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petscfe.h>

/*
  residual = int( phi * f0 ) + int( grad(phi) * f1 )
 */
static void f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal body_force = 0;
  f0[0] = -body_force;
}

static void f1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/*
  The f0 term contains no use of u and so has no contribution to the
  Jacobian. Since our f1 term uses gradient information from u, we
  only have 1 contribution in the jacobian:

  jacobian = int( grad(phi) * g3 * grad(psi) )

  For total generality, g3 is returned as a dim x dim tensor per
  component. In our case we need this to be the identity and so we set
  the diagonal to 1.
 */
static void g3(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d*dim+d] = 1;
}

static PetscErrorCode dirichlet_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[], void *ctx)
{
  bcval[0] = 1.234;
  return 0;
}

int main(int argc,char **argv)
{
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  /* Initialize mesh */
  DM        dm;
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* Initialize the finite element space */
  PetscFE   fe;
  PetscCall(DMCreateFEDefault(dm, 1, NULL, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));

  /* Setup the discrete system */
  PetscDS   ds;
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0, f1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3));

  /* Setup boundary conditions */
  DMLabel   label;
  const int id = 1;
  PetscCall(DMCreateLabel(dm,"marker"));
  PetscCall(DMGetLabel(dm,"marker",&label));
  PetscCall(DMPlexMarkBoundaryFaces(dm,1,label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "all", label, 1, &id, 0, 0, NULL, (void (*)(void)) dirichlet_bc, NULL, NULL, NULL));

  /* Create the nonlinear solver */
  SNES      snes;
  Vec       u;
  PetscReal error;
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, NULL, NULL, NULL));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, u));

  /* Check the error */
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar [],void *) = {dirichlet_bc};
  PetscCall(DMComputeL2Diff(dm, 0, funcs, NULL, u, &error));
  if (PetscAbsReal(error) > 100.0 * PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error has exceeded tolerances: %e\n", error));
  }else{
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error acceptably close to 0\n"));
  }

  /* Cleanup */
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 testset:
   args: -pc_type lu -ksp_type preonly -snes_error_if_not_converged -petscspace_degree 1

   test:
     suffix: tri_p1_patch

   test:
     suffix: quad_p1_patch
     args: -dm_plex_simplex 0

   test:
     suffix: tet_p1_patch
     args: -dm_plex_dim 3 -dm_refine 1

   test:
     suffix: hex_p1_patch
     args: -dm_plex_simplex 0 -dm_plex_dim 3 -dm_refine 1

   test:
     suffix: prism_p1_patch
     args: -dm_plex_cell triangular_prism -dm_plex_dim 3 -dm_refine 1

TEST*/
