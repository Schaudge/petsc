#include <petsc/private/riemannsolverimpl.h>
PetscFunctionList RiemannSolverList              = NULL;
PetscClassId      RIEMANNSOLVER_CLASSID = 0; 


/*@C
  RiemannSolverSetType - Sets the method to be used as the Riemann solver. 

  Collective on RiemannSolver 

  Input Parameters:
+ rs   - The RiemannSolver context
- type - A known method

  Options Database Command:
. -rs_type <type> - Sets the method; use -help for a list of available methods (for instance, lax)

   Notes:
   See "petsc/include/petscriemannsolver.h" for available methods (for instance)
+  RIEMANNLAXFRIEDRICH - lax-friedrich 

   Normally, it is best to use the RiemannSolverSetFromOptions() command and
   then set the RiemannSolver type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the many different solvers.
   The RiemannSolverSetType() routine is provided for those situations where it
   is necessary to set the timestepping solver independently of the
   command line or options database.  This might be the case, for example,
   when the choice of solver changes during the execution of the
   program, and the user's application is taking responsibility for
   choosing the appropriate method.  In other words, this routine is
   not for beginners.

   Level: intermediate

.seealso: RiemannSolver, RiemannSolverCreate(), RiemannSolverSetFromOptions(), RiemannSolverDestroy(), RiemannSolverType

@*/
PetscErrorCode  RiemannSolverSetType(RiemannSolver rs,RiemannSolverType type)
{
  PetscErrorCode (*r)(RiemannSolver);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, RIEMANNSOLVER_CLASSID,1);
  PetscValidCharPointer(type,2);
  ierr = PetscObjectTypeCompare((PetscObject) rs, type, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(RiemannSolverList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown RiemannSolver type: %s", type);
  ierr = RiemannSolverReset(rs);CHKERRQ(ierr);
  if (rs->ops->destroy) {
    ierr = (*(rs)->ops->destroy)(rs);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(rs->ops,sizeof(*rs->ops));CHKERRQ(ierr);
  rs->setupcalled = PETSC_FALSE;
  ierr = PetscObjectChangeTypeName((PetscObject)rs, type);CHKERRQ(ierr);
  ierr = (*r)(rs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*@C
  RiemannSolverGetType - Gets the RiemannSolver method type (as a string).

  Not Collective

  Input Parameter:
. rs - The RiemannSolver

  Output Parameter:
. type - The name of RiemannSolver method

  Level: intermediate

.seealso RiemannSolverSetType()
@*/
PetscErrorCode  RiemannSolverGetType(RiemannSolver rs, RiemannSolverType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,RIEMANNSOLVER_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)rs)->type_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  RiemannSolverRegister - Adds a creation method to the RiemannSolver package.

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  RiemannSolverRegister() may be called multiple times to add several user-defined tses.

  Sample usage:
.vb
  RiemannSolverRegister("my_riemannsolver",  MyRiemannSolverCreate);
.ve

  Then, your riemannsolver type can be chosen with the procedural interface via
.vb
    RiemannSolver rs;
    RiemannSolverCreate(MPI_Comm, &rs);
    RiemannSolverSetType(rs, "my_riemannsolver")
.ve
  or at runtime via the option
.vb
    -rs_type my_riemannsolver 
.ve

  Level: advanced

.seealso: RiemannSolverRegisterAll(), RiemannSolverRegisterDestroy()
@*/
PetscErrorCode  RiemannSolverRegister(const char sname[], PetscErrorCode (*function)(RiemannSolver))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = RiemannSolverInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&RiemannSolverList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

