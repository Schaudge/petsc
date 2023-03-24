#include <petsc/private/riemannsolverimpl.h> /*I "petscriemannsolver.h"  I*/
/*@C
  RiemannSolverCreate - This function creates an empty RiemannSolver. The type of solver 
    can then be set with RiemannSolverSetType().

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. rs   - The RiemannSolver

  Level: beginner

.seealso: RiemannSolverSetType(), RiemannSolverSetUp(), RiemannSolverDestroy()
@*/
PetscErrorCode RiemannSolverCreate(MPI_Comm comm, RiemannSolver *rs)
{
  RiemannSolver r;

  PetscFunctionBegin;
  PetscValidPointer(rs, 1);
  *rs = NULL;
  PetscCall(RiemannSolverInitializePackage());

  PetscCall(PetscHeaderCreate(r, RIEMANNSOLVER_CLASSID, "RiemannSolver", "Riemann Solver", "RiemannSolver", comm, RiemannSolverDestroy, RiemannSolverView));

  /* Add default behavior for RiemannSolver here */
  r->numfields       = -1;
  r->dim             = -1;
  r->fluxfun         = PETSC_NULL;
  r->flux_wrk        = PETSC_NULL;
  r->roeavg          = PETSC_NULL;
  r->computeroemat   = PETSC_NULL;
  r->computeeigbasis = PETSC_NULL;
  r->fluxfunconvex   = PETSC_TRUE; /* For now we will assume the flux function is convex unless told otherwise. This is not a safe option 
  but convenient for now. Should be adjusted as I add more riemann solvers / complicated tests. */
  r->computeroemat   = PETSC_NULL; /* Used in the logic to generate roe matrix data structures, do not edit */
  *rs                = r;
  PetscFunctionReturn(0);
}
