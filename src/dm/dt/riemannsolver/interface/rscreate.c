#include <petsc/private/riemannsolverimpl.h> 
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
PetscErrorCode  RiemannSolverCreate(MPI_Comm comm, RiemannSolver *rs)
{
  RiemannSolver  r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(rs,1);
  *rs = NULL;
  ierr = RiemannSolverInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(r, RIEMANNSOLVER_CLASSID, "RiemannSolver", "Riemann Solver", "RiemannSolver", comm, RiemannSolverDestroy, RiemannSolverView);CHKERRQ(ierr);

  /* Add default behavior for RiemannSolver here */
  r->numfields = -1; 
  r->dim       = -1; 
  r->fluxfun   = PETSC_NULL;
  r->flux_wrk  = PETSC_NULL;
  *rs = r;
  PetscFunctionReturn(0);
}
