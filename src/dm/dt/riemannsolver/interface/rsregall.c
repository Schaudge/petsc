
#include  <petsc/private/riemannsolverimpl.h>
PetscBool         RiemannSolverRegisterAllCalled = PETSC_FALSE;

/* Add Creation Routines here */
PETSC_EXTERN PetscErrorCode RiemannSolverCreate_Lax(RiemannSolver);

/*@C
  RiemannSolverRegisterAll - Registers all of the Riemann solvers in the RiemanSolver package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso: RiemannSolverCreate(), RiemannSolverRegister(), RiemannSolverRegisterDestroy()
@*/
PetscErrorCode  RiemannSolverRegisterAll(void)
{
  PetscFunctionBegin;
  if (RiemannSolverRegisterAllCalled) PetscFunctionReturn(0);
  RiemannSolverRegisterAllCalled = PETSC_TRUE;

  PetscCall(RiemannSolverRegister(RIEMANNLAXFRIEDRICH, RiemannSolverCreate_Lax));

  PetscFunctionReturn(0);
}
