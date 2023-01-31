static const char help[] = "Test the Roe Matrix provided to the RiemannSolver";

/* 
   
*/
#include "../dgnet.h"
#include <petscriemannsolver.h>
#include "../physics.h"

int main(int argc, char *argv[])
{
  char              physname[256] = "shallow";
  PetscFunctionList physics       = 0;
  MPI_Comm          comm          = PETSC_COMM_SELF;
  PetscErrorCode    ierr;
  DGNetwork         dgnet; /* temporarily here as I set up my physics interfaces poorly */
  RiemannSolver     rs;
  PetscInt          numvalues = 10, field, i;
  PetscReal       **uL, **uR, tol = 1e-10;
  PetscRandom      *random;
  PetscViewer       viewer;

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  PetscCall(PetscCalloc1(1, &dgnet));
  /* Register physical models to be available on the command line 
       If testing new physics just add your creation function here. */
  PetscCall(PetscFunctionListAdd(&physics, "shallow", PhysicsCreate_Shallow));

  /* Command Line Options */
  ierr = PetscOptionsBegin(comm, NULL, "Riemann Solver Ex2 Tests Options", "");
  CHKERRQ(ierr);
  PetscCall(PetscOptionsFList("-physics", "Name of physics model to use", "", physics, physname, physname, sizeof(physname), NULL));
  PetscCall(PetscOptionsInt("-numpoints", "Number of random points to check the decomposition at", "", numvalues, &numvalues, NULL));
  PetscCall(PetscOptionsReal("-tol", "Tolerance for comparing the Eigenvectors", "", tol, &tol, NULL));
  ierr = PetscOptionsEnd();
  CHKERRQ(ierr);

  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    PetscCall(PetscFunctionListFind(physics, physname, &r));
    if (!r) SETERRQ1(PETSC_COMM_SELF, 1, "Physics '%s' not found", physname);
    /* Create the physics, will set the number of fields and their names */
    PetscCall((*r)(dgnet));
  }

  /* Set up Riemann Solver */
  PetscCall(RiemannSolverCreate(comm, &rs));
  PetscCall(RiemannSolverSetApplicationContext(rs, dgnet->physics.user));
  PetscCall(RiemannSolverSetFlux(rs, 1, dgnet->physics.dof, dgnet->physics.flux2));
  PetscCall(RiemannSolverSetFluxEig(rs, dgnet->physics.fluxeig));
  PetscCall(RiemannSolverSetJacobian(rs, dgnet->physics.fluxder));
  PetscCall(RiemannSolverSetEigBasis(rs, dgnet->physics.eigbasis));
  PetscCall(RiemannSolverSetRoeMatrixFunct(rs, dgnet->physics.roemat));
  PetscCall(RiemannSolverSetFromOptions(rs));
  PetscCall(RiemannSolverSetUp(rs));

  /* Diagnostic Test */
  /* Generate the random number generators to test the eigendecomposition at. Different field variables 
       may have different relevant values (i.e. non-negative pressure/heigh in Euler/SWE) */
  PetscCall(PetscMalloc1(dgnet->physics.dof, &random));
  for (field = 0; field < dgnet->physics.dof; field++) {
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &random[field]));
    PetscCall(PetscRandomSetInterval(random[field], dgnet->physics.lowbound[field], dgnet->physics.upbound[field])); /* Set a reproducible field for each field variable to generate different numbers for each field */
    PetscCall(PetscRandomSetSeed(random[field], 0x12345678 + 76543 * field));
    PetscCall(PetscRandomSetFromOptions(random[field]));
  }
  /* make the points to test at */
  PetscCall(PetscCalloc2(numvalues, &uL, numvalues, &uR));
  for (i = 0; i < numvalues; i++) {
    PetscCall(PetscCalloc2(dgnet->physics.dof, &uL[i], dgnet->physics.dof, &uR[i]));
    for (field = 0; field < dgnet->physics.dof; field++) {
      PetscCall(PetscRandomGetValueReal(random[field], &uL[i][field]));
      PetscCall(PetscRandomGetValueReal(random[field], &uR[i][field]));
    }
  }
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerSetFromOptions(viewer));

  PetscCall(RiemannSolverTestRoeMat(rs, numvalues, (const PetscReal **)uL, (const PetscReal **)uR, tol, NULL, NULL, viewer));
  /* Print the Failures to the Screen */

  for (i = 0; i < numvalues; i++) { PetscCall(PetscFree2(uL[i], uR[i])); }
  for (field = 0; field < dgnet->physics.dof; field++) { PetscCall(PetscRandomDestroy(&random[field])); }
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFree2(uL, uR));
  PetscCall(PetscFree(random));
  PetscCall(DGNetworkDestroyPhysics(dgnet));
  PetscCall(PetscFree(dgnet));
  PetscCall(RiemannSolverDestroy(&rs));
  PetscCall(PetscFunctionListDestroy(&physics));
  PetscCall(PetscFinalize(); CHKERRQ(ierr));
  return 0;
}
